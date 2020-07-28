# CSN
Compositional Sparse Network on Semantic Segmentation Experiments.

We implement three models : Deeplab V3+, Autodeeplab and CSN. 

DeepLab V3+ is modified from this repository : https://github.com/jfzhang95/pytorch-deeplab-xception

Autodeeplab is modified from this repository : https://github.com/NoamRosenberg/autodeeplab 

CSN is used a backbone of Deeplab V3+ and is also experimented on CIFAR. CSN's deeplab backbone code is a jupyter notebook inside deeplab folder. Whereas, CIFAR-10 CSN notebook is avaliable in parent folder. 

Saltmarsh Dataset can be downloaded from here : https://drive.google.com/drive/folders/1jCAgLq0iI88VTwv6_L4nV0fYGrMLJ0PP?usp=sharing
And to run above scripts, you must put the dataset in Data folder for both deeplab and autodeeplab models. 

## Abstract 

In this paper, we seek inspiration from the fields of neuroscience and continual learning to create networks that can compose meaningful features. We identify three critical principles lacking in modern neural networks that should be part of general-purpose networks: (a) interconnection richness, (b) sparsity, and (c) dynamic expansion. These principles form the basis of a general-purpose network architecture which we term as a \textit{Compositional Sparse Network} (CSN). Since we dynamically expand the network and use learned weights from lower levels to train higher levels in the CSN, the CSN design process can be viewed as a pseudo Neural Architecture Search (NAS) procedure. The CSN is first tested on the CIFAR-10 image data set, and subsequently on the salt marsh image data set where the CSN is used as a backbone for the DeepLab-V3 architecture. We compare the performance of the CSN with an NAS-based approach called Auto-DeepLab which serves as a backbone for the DeepLab-V3 architecture in the context of semantic segmentation of salt marsh images. The proposed CSN approach is observed to perform worse than the NAS-based approach because the higher-level CSNs are seen to not fit the data distribution.
