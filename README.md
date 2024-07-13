
## MutDet: Mutually Optimizing Pre-training for Remote Sensing Object Detection
[Paper][Citing][Appendix] (under construction)

Welcome to the official repository of [MutDet](https://arxiv.org/abs/2103.16607). 
In this work, we propose a pre-training method for object detection in remote sensing images, which can be applied to any DETR-based detector and theoretically extended to other single-stage or two-stage detectors.
Our paper is accepted by ECCV 2024. 

![diagram](.github/images/MutDet_Framework.png)



```
The Arxiv link
```



### Preparation

Please install Python dependencies Following [ARS-DETR](https://github.com/httle/ARS-DETR):


### Datasets

We will release it later.

### The MutDet Framework

Our pre-training framework consists of three steps: 
1. Pseudo-label generation: Using SAM to generate pseudo-boxes and extracting object embeddings using a pre-trained model. 
2. Detection Pre-training: Keeping the backbone frozen and conducting detection pre-training. 
3. Fine-tuning: Fine-tuning on downstream data.

#### 1. Pseudo-label generation

First, we use pre-trained SAM to generate a large number of masks for each image, and then convert these masks into rotated boxes using the minimum bounding box algorithm: 
```shell
bash ./tools/sam_prediction.py
```
Then, we use pre-trained ResNet50 on ImageNet to extract object embeddings:
```shell
bash ./tools/extrat_embeddings.py
```
#### 2. Detection Pre-training
The following command is used for pre-training:
```shell
python ./train.py
```

#### 3. Fine-tuning 
The following command is used for pre-training:
```shell
python ./fine-tuning.py
```
Checkpoints retained during the pre-training process can be directly used to initialize the detector. During initialization, warnings such as 'parameter mismatch' may occur, which is due to MutDet introducing additional modules and using a 256-dimensional classification head. However, the remaining parameters of the detector can be inherited normally, thus not affecting the pre-training effectiveness.

### Results on DOTA and DIOR

![diagram](.github/images/Results_on_DIOR_DOTA.png)



### Pre-trained Models
We will release it later.