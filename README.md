# Introduction
We release the entire code (both training phase & testing phase) for finetuning I3D model on UCF101.   
I3D paper:[Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](http://openaccess.thecvf.com/content_cvpr_2017/papers/Carreira_Quo_Vadis_Action_CVPR_2017_paper.pdf). 
Please also refer to [kinetics-i3d](https://github.com/deepmind/kinetics-i3d) for models and details about I3D.
# Prerequisites
## Software
* Ubuntu 16.04.3 LTS
* Python 2.7
* CUDA8
* CuDNN v6
* Tensorflow 1.4.1
* [Sonnet](https://github.com/deepmind/sonnet)

## Hardware
GTX 1080 Ti

# How to run
## 1. Clone this repo
```bash
git clone https://github.com/USTC-Video-Understanding/I3D_Finetune
```
## 2. Download kinetics pretrained I3D models
In order to finetune I3D network on UCF101, you have to download Kinetics pretrained I3D models provided by DeepMind at [here](https://github.com/deepmind/kinetics-i3d/tree/master/data). Specifically, download the repo [kinetics-i3d](https://github.com/deepmind/kinetics-i3d) and put the `data/checkpoints` folder into `data` subdir of our `I3D_Finetune` repo:
```bash
git clone https://github.com/deepmind/kinetics-i3d
cp -r kinetics-i3d/data/checkpoints I3D_Finetune/data
```
## 3. Create list files
We use list files in `data/ucf101/` subdir to make the code find RGB images and flow data saved on disk. You have to adapt the list files to make sure the list files contain the right path to your data. Specifically, for RGB data, you have to update `data/ucf101/rgb.txt`. Each line in in this file should be in the format:
```bash
dir_name_of_imgs_of_a_video /path/to/img_dir num_imgs label
```
For example, if your RGB data of UCF101 is saved in '/data/user/ucf101/rgb', and there are 13320 subdirs in this folder, each subdir contains images from a video. If in subdir `v_BalanceBeam_g14_c02`, there are 96 images, and the ground truth of this video is 4, then the line for this subdir is:
```bash
v_BalanceBeam_g14_c02 /data/user/ucf101/rgb/v_BalanceBeam_g14_c02 96 4
```
Similarly, update `data/ucf101/flow.txt` for flow data. **Note: we use one file to include x and y part of flow data, so we use `{：s}` in each line to placehold `x` or `y` in the data path.** For example, if your flow data are placed like this:
```bash
|---tvl1_flow
|   |---x
|   |--- y
```
then you can write each line in `flow.txt` like this:
```bash
v_Archery_g01_c06 /data4/zhouhao/dataset/ucf101/tvl1_flow/{:s}/v_Archery_g01_c06 107 2
```
i.e, use `{:s}` replace `x` or `y` in path. If you are confused, please refer our code to see data loading details.

## 4. Train on UCF101 on RGB data and flow data
```bash
# Finetune on split1 of RGB data of UCF101
CUDA_VISIBLE_DEVICES=0 python finetune.py ucf101 rgb 1
# Finetune on split2 of flow data of UCF101
CUDA_VISIBLE_DEVICES=0 python finetune.py ucf101 flow 2 
```
We share our trained models on UCF101(RGB & FLOW) in [GoogleDrive](https://drive.google.com/open?id=1URkdw76Q2yfetDJLPv--2VxWcOg2Q6Hd) and [BaiduDisk](https://pan.baidu.com/s/1LDOlxCfnyZ-EQ4pPwqz5-g) (password:ddar). You can download these models and put them in `model` folder of this repo. In this way you can skip the train commands above and directly run test in the next step.

## 5. test on UCF101 on RGB data and flow data
After you have trained the model, you can run the test procedure. 
First, please update `_DATA_ROOT` and `_CHECKPOINT_PATHS` in `test.py` by setting the value to right location of your dataset and your trained model generated in the previous step, respectively.  
Then you can run testing using below commands:
```bash
# run testing on the split1 of RGB data of UCF101 
CUDA_VISIBLE_DEVICES=0 python test.py ucf101 rgb 1
# run testing on the split1 of flow data of UCF101
CUDA_VISIBLE_DEVICES=0 python test.py ucf101 flow 1
# run testing both on RGB and flow data of split1 of UCF101
CUDA_VISIBLE_DEVICES=0 python test.py ucf101 mixed 1
```

# Results
Our training results on UCF-101 Split-1 are:  

Training Split |      RGB     |    Flow  | Fusion
-------------- | ------------ | ---------|----------
   Split1      |     94.7%     |    96.3%  | **97.6%**

Thanks to tf.Dataset API, we can achieve training speed at 1s/batch(64 frames)!

# Contact
This work is mainly done by Hao Zhou ([Rhythmblue](https://github.com/Rhythmblue)) and Hezhen Hu ([AlexHu123](https://github.com/AlexHu123)). If you have any questions, please create an issue in this repo. We are very happy to hear from you!