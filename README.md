# DFAD

This repository contains the code of paper: *Data-Free Adversarial Distillation*

## Requirements

```bash
pip install -r requirements.txt 
```

## Quick Start: MNIST

This is an MNIST example for DFAD, which only takes a few minutes for training. Data will be automatically downloaded by the python scripts.

```bash
bash run_mnist.sh
```

or 

```bash
# Train the teacher model
python train_teacher.py --batch_size 256 --epochs 10 --lr 0.01 --dataset mnist --model lenet5 --weight_decay 1e-4 # --verbose

# Train the student model
python DFAD_mnist.py --ckpt checkpoint/teacher/mnist-lenet5.pt # --verbose
```

## Step by Step


## 1. Prepare Datasets

MNIST, CIFAR10 and CIFAR100 will be automatically downloaded by the training scripts.  
Download other datasets from the following links and extract them to *./data*:

#### Caltech101 

1. Download [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101) and extract it to *./data/caltech101*
2. Split datasets
    ```bash
    cd data
    python split_caltech101.py
    ```

#### CamVid

1. Download [CamVid](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid) and extract it to *./data/CamVid*

#### NYUv2

1. Download [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and extract it to *./data/NYUv2*
2. Download [labels](https://github.com/ankurhanda/nyuv2-meta-data) and extract it to *./data/NYUv2/nyuv2-meta-data*


## 2. Train teachers and students

Start the visdom server at port 15550 for visualization
.
```bash
visdom -p 15550
```

You can download our pretrained models and extract them to *checkpoint/teacher*. 


### CIFAR

* CIFAR10

```bash
# Teacher
python train_teacher.py --dataset cifar10 --batch_size 128 --step_size 80 --epochs 200 --model resnet34_8x

# Student
python DFAD_cifar.py --dataset cifar10 --ckpt checkpoint/teacher/cifar10-resnet34_8x.pt --scheduler
```

* CIFAR100

```bash
# Teacher
python train_teacher.py --dataset cifar100 --batch_size 128 --step_size 80 --epochs 200 --model resnet34_8x

# Student
python DFAD_cifar.py --dataset cifar100 --ckpt checkpoint/teacher/cifar100-resnet34_8x.pt --scheduler
```

#### Caltech101

```bash
# Teacher 
python train_teacher.py --dataset caltech101 --batch_size 128 --num_classes 101 --step_size 50 --epochs 150 --model resnet34

# Student
python DFAD_caltech101.py --lr_S 0.05 --lr_G 1e-3 --scheduler --batch_size 64 --ckpt checkpoint/teacher/caltech101-resnet34.pt
```

#### CamVid

```bash
# Teacher
python train_teacher_seg.py --model deeplabv3_resnet50 --dataset camvid --data_root ./data/CamVid --scheduler --lr 0.1 --num_classes 11

# Student
python DFAD_camvid_deeplab.py --ckpt checkpoint/teacher/camvid-deeplabv3_resnet50.pt --data_root ./data/CamVid --scheduler
```

#### NYUv2
```bash
# Teacher
python train_teacher_seg.py --model deeplabv3_resnet50 --dataset nyuv2 --data_root ./data/NYUv2 --scheduler --lr 0.05 --num_classes 13

# Student
python DFAD_nyu_deeplab.py --ckpt checkpoint/teacher/nyuv2-deeplabv3_resnet50.pt --data_root ./data/NYUv2 --scheduler
```


