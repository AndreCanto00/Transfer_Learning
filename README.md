# Transfer_Learning

[![Run Python Tests](https://github.com/AndreCanto00/Transfer_Learning/actions/workflows/test.yml/badge.svg)](https://github.com/AndreCanto00/Transfer_Learning/actions/workflows/test.yml)

# Transfer learning-based image classification

**Abstract**: *The aim of the project is to try to overcome the challenge of limited data availability. For this reason has been used the method of transfer learning with the goal of improving classification performance. The target dataset is EuroSAT, a popular resource for Earth observation. The models have been pretrained on miniImageNet, a dataset commonly used in few-shot learning tasks. The project begins by processing the miniImageNet dataset with two distinct architectures that have been employed, ResNet18 and a Vision Transformer. The pre-trained models are evaluated and tested on the validation and test sets of miniImageNet. Subsequently, a small amount of images from EuroSAT has been selected to fine-tune the pretrain models and test them. To ensure ro-
bustness, fine-tuning is performed multiple times, and average results are reported. The report presents a comparative analysis of different models, including ResNet18 and Vision Transformer, evaluating their performance in the context of small datasets.*


**Datasets:**
[miniImageNet](https://drive.google.com/drive/folders/17a09kkqVivZQFggCw9I_YboJ23tcexNM), [EuroSAT(RGB)](https://github.com/phelber/EuroSAT)

## Track:

1. Motivation: In some application domains, we cannot get a large amount of data, which makes it difficult or even impossible to train the deep learning models from scratch. One common approach to address this problem is transfer learning. Researchers attempt to address this problem with transfer learning. * Understand what is transfer learning, and why transfer learning can help to address this problem. 
2. Goal: Improve the performance in remote sensing application with small dataset via transfer learning.

#### Requirements:

1. Complete the project and submit the code. For the code, you can get help from github. (20 points)

    a) Students should try their best to improve the classification performance on EuroSAT by using some strategies like data augmentation. (You can follow these steps to complete the project)

    - i. Download and read the miniImageNet (Because the dataset miniImageNet usually is used in the few-shot learning task. Hence, the train, val, and test data are from different categories. For easier, you can only focus on the train.tar, and split the data in train.tar as train, val, and test dataset, which means you can ignore both val.tar and test.tar.) & EuroSAT(RGB) datasets. (2 points)

    - ii. Pretrain a model (ResNet10, also can be ResNet18, VGG, Vision Transformer, etc.) on the training set of miniImageNet, evaluate & test it on the validation & test set. (7 points)

    - iii. Save the pretrained model. (1 point)

    - iv. Choose 100 images from EuroSAT dataset, which are from 5 different categories and each category includes 20 samples. You should randomly choose 25 images from these 100 samples as training set (The 25 images should be from the 5 different categories. Each category includes 5 images). (3 points)

    - v. Fine-tune the pretrained model with these 25 training images and test it on the rest 75 samples, show the results. Better to fine-tuning several times on different 100 EuroSAT images and get their average result. (7 points)

    b) Compare the performance of different models (ResNet18, VGG, Vision Transformer, etc.), and investigate the different optimization strategies. (This part will be used as a bonus item, 2 points)

    c) If it’s possible, you can evaluate on the other datasets such as CropDiseases, CUB, ISIC, ChestX, etc. (This part will be used as a bonus item, 2 points)

    d) Experiments: python=3.9.18, torch=2.1.1, torchvision=0.16.1

## [Paper here](/transfer_learning-based_image_classification-paper.pdf)

## [Notebook here](/files/)

## How to run:
- Download the Datasets [miniImageNet (train)](https://drive.google.com/drive/folders/17a09kkqVivZQFggCw9I_YboJ23tcexNM), [EuroSAT(RGB)](https://github.com/phelber/EuroSAT). 
- Extract both zip datasets.
- Change the path in the import section of the notebook.
- Now you should be able to run all the notebook.

<br>
<hr>
<br><br>

<div style="text-align: right; font-size: 18px">

*Dec 2023*<br>
**Davide Moricoli, Andrea Cantore, Manex Sorarrain Agirrezabala**
</div>
