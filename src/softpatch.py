import logging
import os
import pickle
import copy
import tqdm
import time
import math
import torch
import torch.nn as nn
import src.common as common
import src.sampler as sampler
import src.multi_variate_gaussian as multi_variate_gaussian
from argparse import ArgumentParser, Namespace
import src.metrics as metrics
from sklearn.neighbors import LocalOutlierFactor
import src.backbones as backbones
from src.fod import FOD
import torch.nn.functional as F
import numpy as np
from src.losses import kl_loss, entropy_loss
import dino.vision_transformer as vits
from info_nce import InfoNCE, info_nce



# from torch_cluster import graclus_cluster

LOGGER = logging.getLogger(__name__)


class SoftPatch(torch.nn.Module):
    def __init__(self, device):
        super(SoftPatch, self).__init__()
        self.device = device
        
    def get_vit_encoder(self, vit_arch, vit_model, vit_patch_size, enc_type_feats):
        if vit_arch == "vit_small" and vit_patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            initial_dim = 384
        elif vit_arch == "vit_small" and vit_patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
            initial_dim = 384
        elif vit_arch == "vit_base" and vit_patch_size == 16:
            if vit_model == "clip":
                url = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
            elif vit_model == "dino":
                url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            initial_dim = 768
        elif vit_arch == "vit_base" and vit_patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            initial_dim = 768

        if vit_model == "dino":
            vit_encoder = vits.__dict__[vit_arch](patch_size=vit_patch_size, num_classes=0)
            # TODO change if want to have last layer not unfrozen
            for p in vit_encoder.parameters():
                p.requires_grad = False
            vit_encoder.eval().cuda()  # mode eval
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/" + url
            )
            vit_encoder.load_state_dict(state_dict, strict=True)

            hook_features = {}
            if enc_type_feats in ["k", "q", "v", "qkv", "mlp"]:
                # Define the hook
                def hook_fn_forward_qkv(module, input, output):
                    hook_features["qkv"] = output

                vit_encoder._modules["blocks"][-1]._modules["attn"]._modules[
                    "qkv"
                ].register_forward_hook(hook_fn_forward_qkv)
        else:
            raise ValueError("Not implemented.")

        return vit_encoder, initial_dim, hook_features

    def load(
        self,
        backbone,
        device,
        input_shape,
        layers_to_extract_from=("layer2", "layer2"),
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        with_fod = False,
        clip_encoder = False,
        refine_contrastive = False,
        cur_class_name=None,
        featuresampler=sampler.ApproximateGreedyCoresetSampler(percentage=0.1, device=torch.device("cuda")),
        nn_method=common.FaissNN(False, 4),
            lof_k=5,
            threshold=0.15,
            weight_method="lof",
            soft_weight_flag=True,
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.with_fod = with_fod # chgd
        self.clip_encoder = clip_encoder
        self.refine_contrastive = refine_contrastive
        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler

        ############SoftPatch ##########
        self.featuresampler = sampler.WeightedGreedyCoresetSampler(featuresampler.percentage,
                                                                   featuresampler.device)
        self.patch_weight = None
        self.feature_shape = []
        self.lof_k = lof_k
        self.threshold = threshold
        self.coreset_weight = None
        self.weight_method = weight_method
        self.soft_weight_flag = soft_weight_flag
        
        if self.clip_encoder:
            '''
            import clip
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, preprocess = clip.load("ViT-B/32", device=device)
            
            self.upsample_1 = nn.Sequential(
                nn.Linear(512, 512*28*28),
                nn.ReLU(),
            ).to(device)
            self.upsample_2 = nn.Sequential(
                nn.Linear(512, 1024*14*14),
                nn.ReLU(),
            ).to(device)
            '''
            vit_arch = 'vit_small'
            vit_model = 'dino'
            self.vit_patch_size = 8
            self.enc_type_feats = 'k'
            
            self.vit_encoder, self.initial_dim, self.hook_features = self.get_vit_encoder(
            vit_arch, vit_model, self.vit_patch_size, self.enc_type_feats)
            
            self.vit_encoder = self.vit_encoder.cuda()
            
            self.auroc_file_name = 'auroc_dict_10n_50e_dino_contrastive.pkl' # pkl
            if os.path.exists(self.auroc_file_name):
                with open(self.auroc_file_name, 'rb') as file:
                    self.final_output_dict = pickle.load(file)
            else:
                self.final_output_dict = {}
        
        
        if refine_contrastive:
            class OneByOneConvNet(nn.Module):
                def __init__(self, in_channels, out_channels):
                    super(OneByOneConvNet, self).__init__()
                    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
                    self.leaky_relu = nn.LeakyReLU(inplace=True)

                def forward(self, x):
                    x = self.conv(x)
                    x = self.leaky_relu(x)
                    return x
    
            self.refine_conv = OneByOneConvNet(384, 1024).cuda()
            
            params = list(self.refine_conv.conv.parameters())
            # params += list(self.upsample_2.parameters())
            
            self.optimizer = torch.optim.Adam(params, lr=1e-3)
            self.CONTRASTIVE_LOSS = InfoNCE()
            # self.IMG_LOSS = nn.CrossEntropyLoss()
            
        
        if self.with_fod:
            models = []
            feat_dims = [512, 1024]
            self.seq_lens = [784, 196]
            self.ws = [28, 14]  # feature map height/width
            args = Namespace(batch_size=8, device=device, with_intra=False, with_inter=True, rfeatures_path='rfeatures_w50',\
                class_name=cur_class_name, feature_levels=len(feat_dims), save_path='checkpoints', save_prefix='mvtec', num_epochs=100,\
                    lr=0.0001, lambda1=0.5, lambda2=0.5) # args chgd here!
            self.args = args
            
            for seq_len, in_channels, d_model in zip(self.seq_lens, feat_dims, [256, 512]):
                model = FOD(seq_len=seq_len,
                            in_channels=in_channels,
                            out_channels=in_channels,
                            d_model=d_model,
                            n_heads=8,
                            n_layers=3,
                            args=args)
                
                print('One Model...Done')
                models.append(model.to(args.device))
                
            self.fod_models = models
            print('Creating Models...Done')
            params = list(models[0].parameters())
            for l in range(1, args.feature_levels):
                params += list(models[l].parameters())  
            self.optimizer = torch.optim.Adam(params, lr=self.args.lr)
            self.l2_criterion = nn.MSELoss()
            self.cos_criterion = nn.CosineSimilarity(dim=-1)
            
            self.auroc_file_name = 'auroc_dict_10n_50e_fod_inter.pkl' # pkl
            
            if os.path.exists(self.auroc_file_name):
                #print("File exists.")
                with open(self.auroc_file_name, 'rb') as file:
                    self.final_output_dict = pickle.load(file)
            else:
                #print("File does not exist.")
                self.final_output_dict = {}
    
    
    @torch.no_grad()
    def extract_feats(self, dims, type_feats="k"):

        nb_im, nh, nb_tokens, _ = dims
        qkv = (
            self.hook_features["qkv"]
            .reshape(
                nb_im, nb_tokens, 3, nh, -1 // nh
            )  # 3 corresponding to |qkv|
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0], qkv[1], qkv[2]

        if type_feats == "q":
            return q.transpose(1, 2).float()
        elif type_feats == "k":
            return k.transpose(1, 2).float()
        elif type_feats == "v":
            return v.transpose(1, 2).float()
        else:
            raise ValueError("Unknown features")
        

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False, with_fod=False, train_steps=None):
        """Returns feature embeddings for images."""
        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        
        
        # CLIP Encoder
        if self.clip_encoder:
            with torch.no_grad():
                w_featmap = images.shape[-2] // self.vit_patch_size 
                h_featmap = images.shape[-1] // self.vit_patch_size 
                att = self.vit_encoder.get_last_selfattention(images) 
            
                # Get decoder features
                feats = self.extract_feats(dims=att.shape, type_feats='k') 
                feats = feats[:, 1:, :, :].reshape(att.shape[0], w_featmap, h_featmap, -1) 
                feats = feats.permute(0, 3, 1, 2) # [B, 384, 28, 28]
                
            if self.refine_contrastive:
                feats = self.refine_conv(feats.clone())
                feats_refine = feats.permute(0,2,3,1).reshape(-1, self.target_embed_dimension)
            
            features = [feats]
                
            #     image_features = self.clip_model.encode_image(images)
            # img_feat1 = self.upsample_1(image_features.to(torch.float32)).reshape(images.shape[0], 512, 28, 28)
            # img_feat2 = self.upsample_2(image_features.to(torch.float32)).reshape(images.shape[0], 1024, 14, 14)
            # features = [img_feat1, img_feat2]
            
        else:
            with torch.no_grad():
                # images: (B, 3, 224, 224)
                features = self.forward_modules["feature_aggregator"](images) # [B, 512, 28, 28]) & [B, 1024, 14, 14]
            
            features = [features[layer] for layer in self.layers_to_extract_from]
        
        # chgd        
        if with_fod:
            path = os.path.join(self.args.save_path, self.args.save_prefix)
            if not os.path.exists(path):
                os.makedirs(path)
            
            loss_rec_list, loss_intra_entropy_list, loss_inter_entropy_list = [], [], []
            loss_corr_list, loss_target_list = [], []
            
            for model in self.fod_models:
                model.train()
            
            for fl in range(self.args.feature_levels): # feature_levels = 2
                m = torch.nn.AvgPool2d(3, 1, 1)
                input = m(features[fl])
                N, D, _, _ = input.shape
                input = input.permute(0, 2, 3, 1).reshape(N, -1, D)

                model = self.fod_models[fl]
                output, intra_corrs, intra_targets, inter_corrs, inter_targets = model(input)
            
                if self.args.with_intra:
                    loss_intra1, loss_intra2, loss_intra_entropy = 0.0, 0.0, 0.0
                    for l in range(len(intra_targets)):
                        L = intra_targets[l].shape[-1]
                        norm_targets = (intra_targets[l] / torch.unsqueeze(torch.sum(intra_targets[l], dim=-1), dim=-1).repeat(1, 1, 1, L)).detach()
                        # optimizing intra correlations
                        loss_intra1 += torch.mean(kl_loss(norm_targets, intra_corrs[l])) + torch.mean(kl_loss(intra_corrs[l], norm_targets))
                        
                        norm_targets = intra_targets[l] / torch.unsqueeze(torch.sum(intra_targets[l], dim=-1), dim=-1).repeat(1, 1, 1, L)
                        loss_intra2 += torch.mean(kl_loss(norm_targets, intra_corrs[l].detach())) + torch.mean(kl_loss(intra_corrs[l].detach(), norm_targets))
                        
                        loss_intra_entropy += torch.mean(entropy_loss(intra_corrs[l]))
                    
                    loss_intra1 = loss_intra1 / len(intra_targets)
                    loss_intra2 = loss_intra2 / len(intra_targets)
                    loss_intra_entropy = loss_intra_entropy / len(intra_targets)

                if self.args.with_inter:
                    loss_inter1, loss_inter2, loss_inter_entropy = 0.0, 0.0, 0.0
                    for l in range(len(inter_targets)):
                        L = inter_targets[l].shape[-1]
                        norm_targets = (inter_targets[l] / torch.unsqueeze(torch.sum(inter_targets[l], dim=-1), dim=-1).repeat(1, 1, 1, L)).detach()
                        # optimizing inter correlations
                        loss_inter1 += torch.mean(kl_loss(norm_targets, inter_corrs[l])) + torch.mean(kl_loss(inter_corrs[l], norm_targets))
                        
                        norm_targets = inter_targets[l] / torch.unsqueeze(torch.sum(inter_targets[l], dim=-1), dim=-1).repeat(1, 1, 1, L)
                        loss_inter2 += torch.mean(kl_loss(norm_targets, inter_corrs[l].detach())) + torch.mean(kl_loss(inter_corrs[l].detach(), norm_targets))

                        loss_inter_entropy += torch.mean(entropy_loss(inter_corrs[l]))
                    
                    loss_inter1 = loss_inter1 / len(inter_targets)
                    loss_inter2 = loss_inter2 / len(inter_targets)
                    loss_inter_entropy = loss_inter_entropy / len(inter_targets)
                
                loss_rec = self.l2_criterion(output, input) + torch.mean(1 - self.cos_criterion(output, input)) # mse + cosine
                
                if self.args.with_intra and self.args.with_inter:  # patch-wise reconstruction + intra correlation + inter correlation
                    loss1 = loss_rec + self.args.lambda1 * loss_intra2 - self.args.lambda1 * loss_inter2 
                    loss2 = loss_rec - self.args.lambda1 * loss_intra1 - self.args.lambda2 * loss_intra_entropy + self.args.lambda1 * loss_inter1 + self.args.lambda2 * loss_inter_entropy 
                elif self.args.with_intra:  # patch-wise reconstruction + intra correlation
                    loss1 = loss_rec + self.args.lambda1 * loss_intra2  
                    loss2 = loss_rec - self.args.lambda1 * loss_intra1 - self.args.lambda2 * loss_intra_entropy 
                elif self.args.with_inter:  # patch-wise reconstruction + inter correlation
                    loss1 = loss_rec - self.args.lambda1 * loss_inter2  
                    loss2 = loss_rec + self.args.lambda1 * loss_inter1 + self.args.lambda2 * loss_inter_entropy 
                else:  # only patch-wise reconstruction
                    loss = loss_rec
                    
                loss_rec_list.append(loss_rec.item())
                if self.args.with_intra and self.args.with_inter:
                    loss_target_list.append((loss_intra2 - loss_inter2).item())
                    loss_corr_list.append((-loss_intra1 + loss_inter1).item())
                    loss_intra_entropy_list.append(loss_intra_entropy.item())
                    loss_inter_entropy_list.append(loss_inter_entropy.item())
                elif self.args.with_intra:
                    loss_target_list.append((loss_intra2).item())
                    loss_corr_list.append((-loss_intra1).item())
                    loss_intra_entropy_list.append(loss_intra_entropy.item())
                elif self.args.with_inter:
                    loss_target_list.append((-loss_inter2).item())
                    loss_corr_list.append((loss_inter1).item())
                    loss_inter_entropy_list.append(loss_inter_entropy.item())

                self.optimizer.zero_grad()
                if not self.args.with_intra and not self.args.with_inter:  # only patch-wise reconstruction
                    loss.backward()
                else:
                    # Two-stage optimization
                    loss1.backward(retain_graph=True)
                    loss2.backward()
                self.optimizer.step()


        # from here is original process
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features # [6, 512, 28, 28] & [6, 1024, 14, 14]
        ]
        
        patch_shapes = [x[1] for x in features] # [6, 784, 512, 3, 3] & [6, 196, 1024, 3, 3]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0] # [[28, 28], [14, 14]]

        for i in range(1, len(features)): # [6, 196, 1024, 3, 3]
            _features = features[i] # 
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        
        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)
        
        
        if with_fod:
            return _detach(features), loss_rec_list, loss_target_list, loss_corr_list, loss_intra_entropy_list, loss_inter_entropy_list
        
        if self.refine_contrastive:
           return _detach(features), feats_refine, patch_shapes
        
        else:
            if provide_patch_shapes:
                return _detach(features), patch_shapes
            return _detach(features)

    def fit(self, training_data, with_fod=False, test_data=None):
        """
        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data, with_fod, test_data)
        

    def _fill_memory_bank(self, input_data, with_fod=False, test_data=None):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image, train_steps=None):
            if with_fod or self.clip_encoder or self.refine_contrastive:
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image, with_fod=with_fod, train_steps=train_steps)
            else:
                with torch.no_grad():
                    input_image = input_image.to(torch.float).to(self.device)
                    return self._embed(input_image, with_fod=with_fod, train_steps=train_steps)

        features = []
        
        if with_fod:
            num_epochs = 50
            self.final_output_dict[input_data.name + '_img_auroc'] = -math.inf
            self.final_output_dict[input_data.name + '_pix_auroc'] = -math.inf
            
            for epoch in range(num_epochs):
                features = []
                with tqdm.tqdm(
                    input_data, desc="Computing support features...", leave=True
                ) as data_iterator:
                    for image in data_iterator:
                        if isinstance(image, dict):
                            image = image["image"]
                        
                        feat, loss_rec_list, loss_target_list, loss_corr_list, \
                            loss_intra_entropy_list, loss_inter_entropy_list = _image_to_features(image, train_steps=len(data_iterator))
                        features.append(feat)
                        
                    print(
                    "Epoch: {0}, Steps: {1} | Rec Loss: {2:.7f} | Target Loss: {3:.7f} | Corr Loss: {4:.7f} | Intra Entropy: {5:.7f} | Inter Entropy: {6:.7f}".format(
                        epoch + 1, len(data_iterator), np.average(loss_rec_list), np.average(loss_target_list), np.average(loss_corr_list), np.average(loss_intra_entropy_list), np.average(loss_inter_entropy_list)))
                    
                    
                features = np.concatenate(features, axis=0)
                with torch.no_grad():
                    self.feature_shape = self._embed(image.to(torch.float).to(self.device), provide_patch_shapes=True)[1][0]
                    patch_weight = self._compute_patch_weight(features)

                    # normalization
                    # patch_weight = (patch_weight - patch_weight.quantile(0.5, dim=1, keepdim=True)).reshape(-1) + 1

                    patch_weight = patch_weight.reshape(-1)
                    threshold = torch.quantile(patch_weight, 1 - self.threshold)
                    sampling_weight = torch.where(patch_weight > threshold, 0, 1)
                    self.featuresampler.set_sampling_weight(sampling_weight)
                    self.patch_weight = patch_weight.clamp(min=0)

                    sample_features, sample_indices = self.featuresampler.run(features)
                    features = sample_features
                    self.coreset_weight = self.patch_weight[sample_indices].cpu().numpy()

                self.anomaly_scorer.fit(detection_features=[features])
                
                # Inference start
                aggregator = {"scores": [], "segmentations": []}
                scores, segmentations, labels_gt, masks_gt = self.predict(test_data)
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)
                scores = np.array(aggregator["scores"])
                min_scores = scores.min(axis=-1).reshape(-1, 1)
                max_scores = scores.max(axis=-1).reshape(-1, 1)
                scores = (scores - min_scores) / (max_scores - min_scores + 1e-5)
                scores = np.mean(scores, axis=0)

                segmentations = np.array(aggregator["segmentations"])
                min_scores = (
                    segmentations.reshape(len(segmentations), -1)
                    .min(axis=-1)
                    .reshape(-1, 1, 1, 1)
                )
                max_scores = (
                    segmentations.reshape(len(segmentations), -1)
                    .max(axis=-1)
                    .reshape(-1, 1, 1, 1)
                )
                segmentations = (segmentations - min_scores) / (max_scores - min_scores)
                segmentations = np.mean(segmentations, axis=0)
                
                LOGGER.info("Computing evaluation metrics.")
                auroc = metrics.compute_imagewise_retrieval_metrics(
                    scores, labels_gt
                )["auroc"]
                
                # Compute PRO score & PW Auroc for all images
                pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                    segmentations, masks_gt
                )
                full_pixel_auroc = pixel_scores["auroc"]
                
                print("image_auroc:", auroc)
                print("pixel_auroc:", full_pixel_auroc)
                print("--------------------------------------")
                
                if auroc > self.final_output_dict[input_data.name + '_img_auroc']:
                    self.final_output_dict[input_data.name + '_img_auroc'] = auroc
                if auroc > self.final_output_dict[input_data.name + '_pix_auroc']:
                    self.final_output_dict[input_data.name + '_pix_auroc'] = full_pixel_auroc
                    
                    # Saving the dictionary to a pickle file
                    with open(self.auroc_file_name, 'wb') as file:
                        pickle.dump(self.final_output_dict, file)
                
            print("Output: ",self.final_output_dict)
        
        elif self.refine_contrastive:
            num_epochs = 50
            self.final_output_dict[input_data.name + '_img_auroc'] = -math.inf
            self.final_output_dict[input_data.name + '_pix_auroc'] = -math.inf
            
            for epoch in range(num_epochs):
                features = []
                refine_features = []
                labels_gt = []
                masks_gt = []
                num_data = []
                num_batches = []
                batch_size = None
                with tqdm.tqdm(
                    input_data, desc="Computing support features...", leave=True
                ) as data_iterator:
                    for idx, image in enumerate(data_iterator):
                        if batch_size == None:
                            batch_size = image['image'].shape[0]
                        labels_gt.extend(image["is_anomaly"].numpy().tolist())
                        masks_gt.extend(image["mask"].numpy().tolist())
                        
                        if isinstance(image, dict):
                            image = image["image"]

                        tmp_img, refine_feats, _ = _image_to_features(image)
                        
                        refine_features.append(refine_feats)
                        features.append(tmp_img)
                        num_data.append(len(features[idx]))
                        num_batches.append(image.shape[0])
                        
                    
                    tmp_img_feats_refine = torch.cat(refine_features).clone()
                    tmp_img_feats = np.concatenate([x.detach().cpu().numpy() for x in refine_features], axis=0)
                    #tmp_img_feats_org = np.concatenate(features, axis=0)
                    
                    #with torch.no_grad():
                    self.feature_shape = self._embed(image.to(torch.float).to(self.device), provide_patch_shapes=True)[2][0]
                    patch_weight = self._compute_patch_weight(tmp_img_feats)
                    patch_weight = patch_weight.reshape(-1)
                    threshold = torch.quantile(patch_weight, 1 - self.threshold)
                    sampling_weight = torch.where(patch_weight > threshold, 0, 1)
                    self.featuresampler.set_sampling_weight(sampling_weight)
                    self.patch_weight = patch_weight.clamp(min=0)
                    sample_features, sample_indices = self.featuresampler.run(tmp_img_feats)
                    tmp_img_feats = sample_features
                    self.coreset_weight = self.patch_weight[sample_indices].cpu().numpy()

                    self.anomaly_scorer.fit(detection_features=[tmp_img_feats])
                
                    num_of_inference = len(data_iterator)
                    org_all_scores = []
                    all_scores = []
                    
                    scores, segmentations, labels_gt, masks_gt = self.predict_train(input_data)         
                    scores = np.array(scores)
                    
                    min_scores = scores.min(axis=-1).reshape(-1, 1)
                    max_scores = scores.max(axis=-1).reshape(-1, 1)
                    scores = (scores - min_scores) / (max_scores - min_scores + 1e-5)
                    scores = np.mean(scores, axis=0)
                    
                    auroc = metrics.compute_imagewise_retrieval_metrics(scores, labels_gt)["auroc"]
                    
                    segmentations = np.array(segmentations)
                    min_scores = (
                        segmentations.reshape(len(segmentations), -1)
                        .min(axis=-1)
                        .reshape(-1, 1, 1, 1)
                    )
                    max_scores = (
                        segmentations.reshape(len(segmentations), -1)
                        .max(axis=-1)
                        .reshape(-1, 1, 1, 1)
                    )
                    segmentations = (segmentations - min_scores) / (max_scores - min_scores)
                    segmentations = np.mean(segmentations, axis=0)
                    
                    pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                    segmentations, masks_gt)["auroc"]
                    print("Img auroc Train: ", auroc)
                    print("Pix auroc Train: ", pixel_scores)
                                        
                    # Loss
                    query = []
                    pos_key = []
                    neg_key = []
                    channel = tmp_img_feats.shape[-1]
                    for idx in range(0, len(num_data)):
                        if idx == len(num_data) - 1:
                            tmp_img = torch.tensor(tmp_img_feats_refine[num_data[idx-1]*idx : num_data[idx-1]*(idx+1) - (num_data[idx-1] - num_data[idx])], requires_grad=True).reshape(-1, self.feature_shape[0], self.feature_shape[1], channel)
                            tmp_score = scores[num_batches[idx-1]*idx : num_batches[idx-1]*(idx+1)- (num_batches[idx-1] - num_batches[idx])]
                        else:
                            tmp_img = torch.tensor(tmp_img_feats_refine[num_data[idx]*idx:num_data[idx]*(idx+1)], requires_grad=True).reshape(batch_size, self.feature_shape[0], self.feature_shape[1], -1)    
                            tmp_score = scores[batch_size*idx:batch_size*(idx+1)]
                        
                        for idx2 in range(len(tmp_score)):
                            if tmp_score[idx2] < 0.5:
                                pos_key.append(tmp_img[idx2].reshape(-1, self.target_embed_dimension))
                            else:
                                neg_key.append(tmp_img[idx2].reshape(-1, self.target_embed_dimension))
                    
                    # Split pos_key
                    half_len = len(pos_key) // 2
                    if half_len % 2 == 0:
                        query = pos_key[:half_len]
                        pos_key = pos_key[half_len:]
                    else:
                        query = pos_key[:half_len]
                        pos_key = pos_key[half_len:]
                        
                    if len(query) > len(pos_key):
                        diff = len(query) - len(pos_key)
                        for _ in range(diff):
                            _ = query.pop(0)
                    
                    elif len(query) < len(pos_key):
                        diff = len(pos_key) - len(query)
                        for _ in range(diff):
                            _ = pos_key.pop(0)
                    
                    #contra_loss = self.CONTRASTIVE_LOSS(torch.cat(query), torch.cat(pos_key), torch.cat(neg_key))
                    contra_loss = self.CONTRASTIVE_LOSS(torch.cat(query)[:50], torch.cat(pos_key)[:50])

                    print(f"contrastive_loss: {contra_loss}")
                    self.optimizer.zero_grad()
                    contra_loss.backward()
                    self.optimizer.step()
                    
                
                # Inference start
                aggregator = {"scores": [], "segmentations": []}
                scores, segmentations, labels_gt, masks_gt = self.predict(test_data)
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)
                scores = np.array(aggregator["scores"])
                min_scores = scores.min(axis=-1).reshape(-1, 1)
                max_scores = scores.max(axis=-1).reshape(-1, 1)
                scores = (scores - min_scores) / (max_scores - min_scores + 1e-5)
                scores = np.mean(scores, axis=0)
                

                segmentations = np.array(aggregator["segmentations"])
                min_scores = (
                    segmentations.reshape(len(segmentations), -1)
                    .min(axis=-1)
                    .reshape(-1, 1, 1, 1)
                )
                max_scores = (
                    segmentations.reshape(len(segmentations), -1)
                    .max(axis=-1)
                    .reshape(-1, 1, 1, 1)
                )
                segmentations = (segmentations - min_scores) / (max_scores - min_scores)
                segmentations = np.mean(segmentations, axis=0)
                
                
                LOGGER.info("Computing evaluation metrics.")
                auroc = metrics.compute_imagewise_retrieval_metrics(
                    scores, labels_gt
                )["auroc"]
                
                # Compute PRO score & PW Auroc for all images
                pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                    segmentations, masks_gt
                )
                full_pixel_auroc = pixel_scores["auroc"]
                
                print("image_auroc:", auroc)
                print("pixel_auroc:", full_pixel_auroc)
                print("--------------------------------------")
                
                if auroc > self.final_output_dict[input_data.name + '_img_auroc']:
                    self.final_output_dict[input_data.name + '_img_auroc'] = auroc
                if auroc > self.final_output_dict[input_data.name + '_pix_auroc']:
                    self.final_output_dict[input_data.name + '_pix_auroc'] = full_pixel_auroc
                    
                    # Saving the dictionary to a pickle file
                    with open(self.auroc_file_name, 'wb') as file:
                        pickle.dump(self.final_output_dict, file)
                
            print("Output: ",self.final_output_dict)
                            
        
        else: # original
            with tqdm.tqdm(
                input_data, desc="Computing support features...", leave=True
            ) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image = image["image"]
                    features.append(_image_to_features(image))
            
            
            features = np.concatenate(features, axis=0) # (163856, 1024)
            with torch.no_grad():
                self.feature_shape = self._embed(image.to(torch.float).to(self.device), provide_patch_shapes=True)[1][0]                
                patch_weight = self._compute_patch_weight(features)

                # normalization
                # patch_weight = (patch_weight - patch_weight.quantile(0.5, dim=1, keepdim=True)).reshape(-1) + 1

                patch_weight = patch_weight.reshape(-1)
                threshold = torch.quantile(patch_weight, 1 - self.threshold)
                sampling_weight = torch.where(patch_weight > threshold, 0, 1)
                self.featuresampler.set_sampling_weight(sampling_weight)
                self.patch_weight = patch_weight.clamp(min=0)

                sample_features, sample_indices = self.featuresampler.run(features)
                features = sample_features
                self.coreset_weight = self.patch_weight[sample_indices].cpu().numpy()

            self.anomaly_scorer.fit(detection_features=[features])

    def _compute_patch_weight(self, features: np.ndarray):
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)

        reduced_features = self.featuresampler._reduce_features(features)
        patch_features = \
            reduced_features.reshape(-1, self.feature_shape[0]*self.feature_shape[1], reduced_features.shape[-1])

        # if aligned:
        #     codebook = patch_features[0]
        #     assign = []
        #     for i in range(1, patch_features.shape[0]):
        #         dist = torch.cdist(codebook, patch_features[i]).cpu().numpy()
        #         row_ind, col_ind = linear_assignment(dist)
        #         assign.append(col_ind)
        #         patch_features[i]=torch.index_select(patch_features[i], 0, torch.from_numpy(col_ind).to(self.device))

        patch_features = patch_features.permute(1, 0, 2)

        if self.weight_method == "lof":
            patch_weight = self._compute_lof(self.lof_k, patch_features).transpose(-1, -2)
        elif self.weight_method == "lof_gpu":
            patch_weight = self._compute_lof_gpu(self.lof_k, patch_features).transpose(-1, -2)
        elif self.weight_method == "nearest":
            patch_weight = self._compute_nearest_distance(patch_features).transpose(-1, -2)
            patch_weight = patch_weight + 1
        elif self.weight_method == "gaussian":
            gaussian = multi_variate_gaussian.MultiVariateGaussian(patch_features.shape[2], patch_features.shape[0])
            stats = gaussian.fit(patch_features)
            patch_weight = self._compute_distance_with_gaussian(patch_features, stats).transpose(-1, -2)
            patch_weight = patch_weight + 1
        else:
            raise ValueError("Unexpected weight method")

        # if aligned:
        #     patch_weight = patch_weight.cpu().numpy()
        #     for i in range(0, patch_weight.shape[0]):
        #         patch_weight[i][assign[i]] = patch_weight[i]
        #     patch_weight = torch.from_numpy(patch_weight).to(self.device)

        return patch_weight

    def _compute_distance_with_gaussian(self, embedding: torch.Tensor, stats: [torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embedding (Tensor): Embedding Vector
            stats (List[Tensor]): Mean and Covariance Matrix of the multivariate Gaussian distribution

        Returns:
            Anomaly score of a test image via mahalanobis distance.
        """
        # patch, batch, channel = embedding.shape
        embedding = embedding.permute(1, 2, 0)

        mean, inv_covariance = stats
        delta = (embedding - mean).permute(2, 0, 1)

        distances = (torch.matmul(delta, inv_covariance) * delta).sum(2)
        distances = torch.sqrt(distances)

        return distances

    def _compute_nearest_distance(self, embedding: torch.Tensor) -> torch.Tensor:
        patch, batch, _ = embedding.shape

        x_x = (embedding ** 2).sum(dim=-1, keepdim=True).expand(patch, batch, batch)
        dist_mat = (x_x + x_x.transpose(-1, -2) - 2 * embedding.matmul(embedding.transpose(-1, -2))).abs() ** 0.5
        nearest_distance = torch.topk(dist_mat, dim=-1, largest=False, k=2)[0].sum(dim=-1)  #
        # nearest_distance = nearest_distance.transpose(0, 1).reshape(batch * patch)
        return nearest_distance

    def _compute_lof(self, k, embedding: torch.Tensor) -> torch.Tensor:
        patch, batch, _ = embedding.shape   # 784x219x128
        clf = LocalOutlierFactor(n_neighbors=int(k), metric='l2')
        scores = torch.zeros(size=(patch, batch), device=embedding.device)
        for i in range(patch):
            if embedding.requires_grad == True:
                clf.fit(embedding[i].cpu().detach().numpy())
            else:
                clf.fit(embedding[i].cpu())
            scores[i] = torch.Tensor(- clf.negative_outlier_factor_)
            # scores[i] = scores[i] / scores[i].mean()   # normalization
        # embedding = embedding.reshape(patch*batch, channel)
        # clf.fit(embedding.cpu())
        # scores = torch.Tensor(- clf.negative_outlier_factor_)
        # scores = scores.reshape(patch, batch)
        return scores

    def _compute_lof_gpu(self, k, embedding: torch.Tensor) -> torch.Tensor:
        """
        GPU support
        """

        patch, batch, _ = embedding.shape

        # calculate distance
        x_x = (embedding ** 2).sum(dim=-1, keepdim=True).expand(patch, batch, batch)
        dist_mat = (x_x + x_x.transpose(-1, -2) - 2 * embedding.matmul(embedding.transpose(-1, -2))).abs() ** 0.5 + 1e-6

        # find neighborhoods
        top_k_distance_mat, top_k_index = torch.topk(dist_mat, dim=-1, largest=False, k=k + 1)
        top_k_distance_mat, top_k_index = top_k_distance_mat[:, :, 1:], top_k_index[:, :, 1:]
        k_distance_value_mat = top_k_distance_mat[:, :, -1]

        # calculate reachability distance
        reach_dist_mat = torch.max(dist_mat, k_distance_value_mat.unsqueeze(2).expand(patch, batch, batch)
                                   .transpose(-1, -2))  # Transposing is important
        top_k_index_hot = torch.zeros(size=dist_mat.shape, device=top_k_index.device).scatter_(-1, top_k_index, 1)

        # Local reachability density
        lrd_mat = k / (top_k_index_hot * reach_dist_mat).sum(dim=-1)

        # calculate local outlier factor
        lof_mat = ((lrd_mat.unsqueeze(2).expand(patch, batch, batch).transpose(-1, -2) * top_k_index_hot).sum(
            dim=-1) / k) / lrd_mat
        return lof_mat


    def _chunk_lof(self, k, embedding: torch.Tensor) -> torch.Tensor:
        width, height, batch, channel = embedding.shape
        chunk_size = 2

        new_width, new_height = int(width / chunk_size), int(height / chunk_size)
        new_patch = new_width * new_height
        new_batch = batch * chunk_size * chunk_size

        split_width = torch.stack(embedding.split(chunk_size, dim=0), dim=0)
        split_height = torch.stack(split_width.split(chunk_size, dim=1 + 1), dim=1)

        new_embedding = split_height.view(new_patch, new_batch, channel)
        lof_mat = self._compute_lof(k, new_embedding)
        chunk_lof_mat = lof_mat.reshape(new_width, new_height, chunk_size, chunk_size, batch)
        chunk_lof_mat = chunk_lof_mat.transpose(1, 2).reshape(width, height, batch)
        return chunk_lof_mat

    
    def predict_train(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_train(data)
        return self._predict_train(data)
    
    def _predict_dataloader_train(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=True) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict_train(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt
    
    
    def _predict_train(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.train()

        batchsize = images.shape[0]
        #with torch.no_grad():
        if self.refine_contrastive:
            features, refine_feats, patch_shapes = self._embed(images, provide_patch_shapes=True)
        
        else:
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
        
        features = np.asarray(features) # (4704, 1024)

        image_scores, _, indices = self.anomaly_scorer.predict([features])
        if self.soft_weight_flag:
            indices = indices.squeeze()
            # indices = torch.tensor(indices).to(self.device)
            weight = np.take(self.coreset_weight, axis=0, indices=indices)

            image_scores = image_scores * weight
            # image_scores = weight

        patch_scores = image_scores

        image_scores = self.patch_maker.unpatch_scores(
            image_scores, batchsize=batchsize
        )
        image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
        image_scores = self.patch_maker.score(image_scores)

        patch_scores = self.patch_maker.unpatch_scores(
            patch_scores, batchsize=batchsize
        )
        scales = patch_shapes[0]
        patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

        masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]
    

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=True) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            if self.refine_contrastive:
                features, refine_feats, patch_shapes = self._embed(images, provide_patch_shapes=True)
            
            else:
                features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            
            features = np.asarray(features) # (4704, 1024)

            image_scores, _, indices = self.anomaly_scorer.predict([features])
            if self.soft_weight_flag:
                indices = indices.squeeze()
                # indices = torch.tensor(indices).to(self.device)
                weight = np.take(self.coreset_weight, axis=0, indices=indices)

                image_scores = image_scores * weight
                # image_scores = weight

            patch_scores = image_scores

            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            params = pickle.load(load_file)
        params["backbone"] = backbones.load(
            params["backbone.name"]
        )
        params["backbone"].name = params["backbone.name"]
        del params["backbone.name"]
        self.load(**params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for side in features.shape[-2:]:
            n_patches = (
                side + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, patch_scores, batchsize):
        return patch_scores.reshape(batchsize, -1, *patch_scores.shape[1:])

    def score(self, image_scores):
        was_numpy = False
        if isinstance(image_scores, np.ndarray):
            was_numpy = True
            image_scores = torch.from_numpy(image_scores)
        while image_scores.ndim > 1:
            image_scores = torch.max(image_scores, dim=-1).values
        if was_numpy:
            return image_scores.numpy()
        return image_scores
