# deeplab on Salt marsh data




### Introduction
We implement Deeplab v3+ with ResNet and CSN backbones on Salt marsh dataset[DeepLab-V3-Plus](https://arxiv.org/pdf/1802.02611).

![Results](doc/2.png)


### Installation
0. Clone the repo:
    ```Shell
    git clone https://github.com/jfzhang95/pytorch-deeplab-xception.git
    cd pytorch-deeplab-xception
    ```

1. Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    For custom dependencies:
    ```Shell
    pip install matplotlib pillow tensorboardX tqdm
    ```
### Training of Deeplab with ResNet backbone
Follow steps below to train your model:

0. Configure your dataset path and download the salt marsh dataset in Data folder from google drive link in parent folder.

1. Input arguments: (see full input arguments via python train.py --help):
    ```Shell
    usage: train.py [-h] [--backbone {resnet,xception,drn,mobilenet}]
                [--out-stride OUT_STRIDE] [--dataset {marsh}]
                [--use-sbd] [--workers N] [--base-size BASE_SIZE]
                [--crop-size CROP_SIZE] [--sync-bn SYNC_BN]
                [--freeze-bn FREEZE_BN] [--loss-type {ce,focal}] [--epochs N]
                [--start_epoch N] [--batch-size N] [--test-batch-size N]
                [--use-balanced-weights] [--lr LR]
                [--lr-scheduler {poly,step,cos}] [--momentum M]
                [--weight-decay M] [--nesterov] [--no-cuda]
                [--gpu-ids GPU_IDS] [--seed S] [--resume RESUME]
                [--checkname CHECKNAME] [--ft] [--eval-interval EVAL_INTERVAL]
                [--no-val]

    ```
  ### Training of Deeplab with Compositional Sparse backbone
 0. Configure your dataset path and download the salt marsh dataset in Data folder from google drive link in parent folder.

 1. Install jupyter notebook and open by ```jupyter DeepLabV3 and CSN onSaltmarsh.ipynb```
 
 2. Run all columns of DeepLabV3 and CSN onSaltmarsh.ipynb
 
 3. Currently, to test out different hyperparameters of CSN, you will have make changes manually. We will update documentation to make this process better later. 


### Acknowledgement
This repository is heavily influenced from [JF Zhang's Deeplab v3+ repo](https://github.com/jfzhang95/pytorch-deeplab-xception).
[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)

Compositional Sparse Network's paper is under progress. 
