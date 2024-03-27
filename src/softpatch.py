import logging
import os
import pickle
import tqdm
import time
import torch
import torch.nn as nn
import src.common as common
import src.sampler as sampler
import src.multi_variate_gaussian as multi_variate_gaussian
from argparse import ArgumentParser, Namespace
from sklearn.neighbors import LocalOutlierFactor
import src.backbones as backbones
from src.fod import FOD
import torch.nn.functional as F
import numpy as np
from src.losses import kl_loss, entropy_loss

# from torch_cluster import graclus_cluster

LOGGER = logging.getLogger(__name__)


class SoftPatch(torch.nn.Module):
    def __init__(self, device):
        super(SoftPatch, self).__init__()
        self.device = device

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
        
        if self.with_fod:
            models = []
            feat_dims = [512, 1024]
            self.seq_lens = [784, 196]
            self.ws = [28, 14]  # feature map height/width
            args = Namespace(batch_size=8, device=device, with_intra=True, with_inter=True, rfeatures_path='rfeatures_w50',\
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
        
        
        with torch.no_grad():
            # images: (8, 3, 224, 224)
            features = self.forward_modules["feature_aggregator"](images) # [8, 512, 28, 28]) & [8, 1024, 14, 14]
            
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
                    # loss1.requires_grad_(True) # chgd
                    # loss2.requires_grad_(True) # chgd
                    
                    loss1.backward(retain_graph=True)
                    loss2.backward()
                self.optimizer.step()


        # from here is original
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
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
            
        else:
            if provide_patch_shapes:
                return _detach(features), patch_shapes
            return _detach(features)

    def fit(self, training_data, with_fod=False):
        """
        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """        
        self._fill_memory_bank(training_data, with_fod)
        

    def _fill_memory_bank(self, input_data, with_fod=False):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image, train_steps=None):
            if with_fod:
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image, with_fod=with_fod, train_steps=train_steps)
            else:
                with torch.no_grad():
                    input_image = input_image.to(torch.float).to(self.device)
                    return self._embed(input_image, with_fod=with_fod, train_steps=train_steps)

        features = []
        
        if with_fod:
            num_epochs = 20
            for epoch in range(num_epochs):
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
                    
                    
        else: # original
            with tqdm.tqdm(
                input_data, desc="Computing support features...", leave=True
            ) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image = image["image"]
                    features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)

        with torch.no_grad():
            # pdb.set_trace()
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
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

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
