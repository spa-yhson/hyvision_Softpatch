# datapath=../../MVTec
datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut'
'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')

dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

python main.py --dataset mvtec --data_path ../data/mvtec --noise 0.1  "${dataset_flags[@]}" --gpu 0 --with_fod --batch_size 6

#python main.py --dataset mvtec --data_path /mnt/sda/yhson/data/mvtec --noise 0.1  "${dataset_flags[@]}" --gpu 0
