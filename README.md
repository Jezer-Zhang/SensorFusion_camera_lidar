# Research Focus
This study aimed to investigate the fusion approaches based on autoencoders for image reconstruction, particularly in their ability to restore missing objects in the images. To achieve this aim, various fusion approaches were designed and tested. These approaches encompassed a combination of the feature representations with **1.scalar weights**, **2.elementwise weights through neural network**, **3.the use of convolution layers for dimension adjustor after concatenating the feature representations** and **4.elementwise weights through mathematical functions**. 
Notably, Fusion 1 and Fusion 4 did not require a retraining, while Fusion 2 and 3 necessitated a retraining in the fusion part of the network.

# Results
The results indicate that sensor fusion via autoencoders can greatly improve the quality of the reconstructed images and help restore the missing objects. However, the models' performance is notably limited when confronted with new types of damages. To tackle this problem, Fusion 4 involving mathematical functions was proposed and achieved good performance.
![Image text](https://github.com/Jezer-Zhang/SensorFusion_camera_lidar/blob/main/results_DeepEnsemble.png)
