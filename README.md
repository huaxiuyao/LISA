# Improving Out-of-Distribution Robustness via Selective Augmentation

This code implements the LISA algorithm.

If you find this repository useful in your research, please cite the following paper:
```
@inproceedings{yao2022improving,
  title={Improving Out-of-Distribution Robustness via Selective Augmentation},
  author={Yao, Huaxiu and Wang, Yu and Li, Sai and Zhang, Linjun and Liang, Weixin and Zou, James and Finn, Chelsea},
  booktitle={Proceeding of the Thirty-ninth International Conference on Machine Learning},
  year={2022}
}
```

The experiments are based on the code:
- [group_DRO](https://github.com/kohpangwei/group_DRO) for subpulation shifts and MetaShifts;
- [fish](https://github.com/YugeTen/fish) for domain shifts and CivilComments;

## Abstract

Machine learning algorithms typically assume that training and test examples are drawn from the
same distribution. However, distribution shift is a common problem in real-world applications and
can cause models to perform dramatically worse at test time. In this paper, we specifically consider
the problems of subpopulation shifts (e.g., imbalanced data) and domain shifts. While prior works
often seek to explicitly regularize internal representations or predictors of the model to be domain
invariant, we instead aim to learn invariant predictors without restricting the model’s internal
representations or predictors. This leads to a simple mixup-based technique which learns invariant
predictors via selective augmentation called LISA.
LISA selectively interpolates samples either with
the same labels but different domains or with the
same domain but different labels. Empirically, we
study the effectiveness of LISA on nine benchmarks ranging from subpopulation shifts to domain shifts, and we find that LISA consistently
outperforms other state-of-the-art methods and
leads to more invariant predictors. We further analyze a linear setting and theoretically show how
LISA leads to a smaller worst-group error.

## Prerequisites
- python 3.6.8
- matplotlib 3.0.3
- numpy 1.16.2
- pandas 0.24.2
- pillow 5.4.1
- pytorch 1.1.0
- pytorch_transformers 1.2.0
- torchvision 0.5.0a0+19315e3
- tqdm 4.32.2
- wilds 2.0.0

## Datasets and Scripts

### Subpopulation shifts and MetaShifts
To run the code, you need to first enter the directory: `cd subpopulation_shifts`. Then change the `root_dir` variable in `./data/data.py` if you need to put the dataset elsewhere other than `./data/`. 

For subpopulation shifts problems, the datasets are listed as follows:


#### MetaShifts
The dataset can be downloaded [[here]](https://drive.google.com/file/d/1Fr2HxUOL3_QUDHU5B3MMH7dgFu_u_gJ_/view?usp=sharing). You should put it under the directory `data`. The running scripts for 4 dataset with different distances are as follows:
```
python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300 --gamma 0.1 --dog_group 1 --lisa_mix_up --mix_alpha 2 --cut_mix --group_by_label
python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300 --gamma 0.1 --dog_group 2 --lisa_mix_up --mix_alpha 2 --cut_mix --group_by_label
python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300 --gamma 0.1 --dog_group 3 --lisa_mix_up --mix_alpha 2 --cut_mix --group_by_label
python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300 --gamma 0.1 --dog_group 4 --lisa_mix_up --mix_alpha 2 --cut_mix --group_by_label
```

#### CMNIST
This dataset is constructed from MNIST. It will be automatically downloaded when running the following script:
```
python run_expt.py -s confounder -d CMNIST -t 0-4 -c isred --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --lisa_mix_up --mix_ratio 0.5`
```

#### CelebA
This dataset can be downloaded via the link in the repo [group_DRO](https://github.com/kohpangwei/group_DRO). 

The command to run LISA on CelebA is:
```
python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.0001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 50 --gamma 0.1 --generalization_adjustment 0 --lisa_mix_up --mix_alpha 2 --mix_ratio 0.5 --cut_mix`
```

#### Waterbirds
This dataset can be downloaded via the link in the repo [group_DRO](https://github.com/kohpangwei/group_DRO). 

The command to run LISA on Waterbirds is:
```
python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --lisa_mix_up --mix_alpha 2 --mix_ratio 0.5`
```



### Domain Shifts
To run the code, you need to first enter the directory: `cd domain_shifts`.

Our implementation and the processing of the datasets are based on the repo [fish](https://github.com/YugeTen/fish). The datasets will be automatically downloaded when running the scripts provided below. 

#### Camelyon17
```
python main.py --dataset camelyon --algorithm lisa --data-dir /data/wangyu/Cameyon17 --group_by_label
```

#### FMoW
```
python main.py --dataset fmow --algorithm lisa --data-dir /data/wangyu/FMoW --group_by_label
```

#### RxRx1
```
python main.py --dataset rxrx --algorithm lisa --data-dir /data/wangyu/RxRx1 --group_by_label
```

#### Amazon
```
python main.py --dataset amazon --algorithm lisa --data-dir /data/wangyu/Amazon --group_by_label
```

#### CivilComments
```
python main.py --dataset civil --algorithm lisa --data-dir /data/wangyu/CivilComments --mix_unit group
```



