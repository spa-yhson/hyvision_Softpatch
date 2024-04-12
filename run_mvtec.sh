# datapath=../../MVTec
datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut'
'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')

dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

#CUDA_VISIBLE_DEVICES=1 python main.py --dataset mvtec --data_path ../data/mvtec --noise 0.1  "${dataset_flags[@]}" --gpu 0 --with_fod --batch_size 6

CUDA_VISIBLE_DEVICES=0 python main.py --dataset mvtec --data_path ../data/mvtec --noise 0.1  "${dataset_flags[@]}" --gpu 0 --clip_encoder --refine_contrastive --batch_size 6

#python main.py --dataset mvtec --data_path ../data/mvtec --noise -1  "${dataset_flags[@]}" --gpu 0 --batch_size 6 
