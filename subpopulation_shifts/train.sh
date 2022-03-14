## MetaDataset

# MetaDataset ERM;  Experiment 1
python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1  --dog_group 1
# MetaDataset ERM;  Experiment 2
python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1  --dog_group 2
# MetaDataset ERM;  Experiment 3
python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1  --dog_group 3
# MetaDataset ERM;  Experiment 4
python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1  --dog_group 4


# MetaDataset Mix up;  Experiment 1
python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1  --dog_group 1 --lisa_mix_up --group_by_label
# MetaDataset CutMix;  Experiment 1
python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1  --dog_group 1 --lisa_mix_up --cut_mix --group_by_label


## CMNIST
# ERM
python run_expt.py -s confounder -d CMNIST -t 0-4 -c isred --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0
# Mixup
python run_expt.py -s confounder -d CMNIST -t 0-4 -c isred --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --lisa_mix_up
# CutMix
python run_expt.py -s confounder -d CMNIST -t 0-4 -c isred --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --lisa_mix_up --cut_mix


## WaterBirds
# ERM
python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0
# Mixup
python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --lisa_mix_up --log_every 5
# CutMix
python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --lisa_mix_up --cut_mix --log_every 5


## CelebA
# ERM
python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.0001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 50 --gamma 0.1 --generalization_adjustment 0
# Mixup; Our 300 epochs are not the same as 300 epochs in ERM.
python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.0001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300 --gamma 0.1 --generalization_adjustment 0 --lisa_mix_up
# CutMix
python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.0001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300 --gamma 0.1 --generalization_adjustment 0 --lisa_mix_up --cut_mix
