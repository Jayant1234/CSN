# CSN
Compositional Sparse Network on Semantic Segmentation Experiments.

We implement three models : Deeplab V3+, Autodeeplab and CSN. 

DeepLab V3+ is modified from this repository : https://github.com/jfzhang95/pytorch-deeplab-xception

Autodeeplab is modified from this repository : https://github.com/NoamRosenberg/autodeeplab 

CSN is used a backbone of Deeplab V3+ and is also experimented on CIFAR. CSN's deeplab backbone code is a jupyter notebook inside deeplab folder. 

Whereas CIFAR-10 CSN notebook is avaliable in parent folder. 

Saltmarsh Dataset can be downloaded from here : https://drive.google.com/drive/folders/1jCAgLq0iI88VTwv6_L4nV0fYGrMLJ0PP?usp=sharing
And to run above scripts, you must put the dataset in Data folder for both deeplab and autodeeplab models. 
