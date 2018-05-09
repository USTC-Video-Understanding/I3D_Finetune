# Introduction
We release the entire code(both train & test) for I3D [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](http://openaccess.thecvf.com/content_cvpr_2017/papers/Carreira_Quo_Vadis_Action_CVPR_2017_paper.pdf). 

# Prerequisites
## Software
Ubuntu 16.04.3 LTS ;Python 2.7
Tensorflow 1.4.1; Sonnet; NumPy
## Hardware
GTX 1080 Ti

# Set up
1. you can train the model on UCF-101 in the following options:
    CUDA_VISIBLE_DEVICES=? python Demo_Transfer.py ucf101 rgb
    CUDA_VISIBLE_DEVICES=? python Demo_Transfer.py ucf101 flow 
    
2. after you have trained the model, you can run the test procedure:
    CUDA_VISIBLE_DEVICES=? python test.py ucf101 rgb
    CUDA_VISIBLE_DEVICES=? python test.py ucf101 flow
    CUDA_VISIBLE_DEVICES=? python test.py ucf101 mixed

3. We share our training results on UCF-101(RGB & FLOW) in [GoogleDrive](https://drive.google.com/open?id=1URkdw76Q2yfetDJLPv--2VxWcOg2Q6Hd) and [BaiduDisk](https://pan.baidu.com/s/1LDOlxCfnyZ-EQ4pPwqz5-g) (password:ddar) output folder, which contains models finetuned on UCF-101.

4. Our training results on UCF-101 Split-1 is:
Training Split |      RGB     |    Flow
-------------- | ------------ | -----------
   Split1      |     94.7     |    96.3

5. Thanks to tf.Dataset API, we can achieve training speed at 1s/batch!

