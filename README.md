# CNN-SPP : ["Convolutional neural network with spatial pyramid pooling for hand gesture recognition"](https://link.springer.com/article/10.1007/s00521-020-05337-0)
Implementation of data augmentation (in python) and CNN-SPP (in tensorflow 1.15) in ["Convolutional neural network with spatial pyramid pooling for hand gesture recognition"](https://link.springer.com/article/10.1007/s00521-020-05337-0)


| CNN-SPP                                                                                              
|---------------------------------------------------------------------------------------------------------
![CNN-SPP](https://github.com/YongSoonTan/CNN-SPP/blob/main/CNN-SPP.png)

if you find this code useful for your research, please consider citing:

    @article{tan2020convolutional,
      title={Convolutional neural network with spatial pyramid pooling for hand gesture recognition},
      author={Tan, Yong Soon and Lim, Kian Ming and Tee, Connie and Lee, Chin Poo and Low, Cheng Yaw},
      journal={Neural Computing and Applications},
      pages={1--13},
      year={2020},
      publisher={Springer}
    }
    
 if GPU memory is not an issue, during testing, you can run all test images at once, just remove for loop in line 281 and line 395, and dedent the block of codes, and set the test images and labels accordingly. 
 
 ## Datasets (All three static hand gesture datasets in raw form ["here"])
 For ASL dataset, augmented training set is not provided, as they are too large to upload (2GB for each fold). However, training set without augmented data is provided, each fold of the training sets is compressed into 3 parts. You can reproduce training sets with augmented data using the Data_Aug.py file provided.
 
 For ASL with digits and NUS hand gesture dataset, training set with and without augmented data are both provided, where augmented training sets are compressed into parts.
 
 Data augmentation needs to be applied to the each fold of the training sets, images are not augmented in real-time during training. 
 To generate augmented data for ASL dataset, please read instruction on line 16 in Data_Aug.py.

| ASL                                                                                              
|---------------------------------------------------------------------------------------------------------
![ASL](https://github.com/YongSoonTan/CNN-SPP/blob/main/ASL.jpg)

| ASL with digits                                                                                               | NUS hand gesture                                            
|---------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------
| ![ASL_with_digits](https://github.com/YongSoonTan/CNN-SPP/blob/main/ASL_with_digits.jpg) | ![NUS](https://github.com/YongSoonTan/CNN-SPP/blob/main/NUS.jpg) |

| Data augmentation                                                                                              
|---------------------------------------------------------------------------------------------------------
![ASL](https://github.com/YongSoonTan/CNN-SPP/blob/main/Data_Augmentation.jpg)

