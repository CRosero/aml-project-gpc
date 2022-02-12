# Adversarial Domain Adaptation for Real-time Semantic Segmentation
## Final Project – "Advanced Machine Learning" 2021 - 2022 Course at PoliTO
### Giulia D'Ascenzi, Patrizio de Girolamo, Carlos Rosero

PyTorch implementation of the algorithm implemented for the final project of the course "Advanced Machine Learning". The assignment of the project can be consulted [here](Assignement.pdf).

*Requirements*: pytorch 0.4.1, python 3.6, torchinfo.


## Datasets

The datasets used are subsets of the Cityscapes and GTA5 datasets. They can be downloaded here: [data.zip](https://drive.google.com/file/d/1Q4yZdjx9WOn7EYU6FlHE9Vpamvpn15L2/view?usp=sharing)

## 1. Fully supervised training with the target dataset.
The file `FS_train.py` can be used to train the network BiSeNet in a supervised way using the target dataset (Cityscapes).

Example:
```
python FS_train.py '--num_epochs', '50',
                    '--data', 'path/to/data/folder'
                            

```

## 2. Unsupervised domain adaptation
The file `DA_train.py` can be used to train the network BiSeNet using unsupervised adversarial learning for the GTA5 -> Cityscapes domain adaptation task.

The file `model/discriminator.py` contains the models of the two implemented discriminators: the Fully Convlutional discriminator and the Light Weight Discriminator. 

The 'architectures' folder describes the architectures of the two discriminators and BiSeNet.

To use the Light Weight version set '--light_discriminator, True'.

In `utils/FDA.py` the Fourier Domain Adaptation transformation is implemented as described in the [original code](https://github.com/YanchaoYang/FDA).

To use FDA set '--FDA, True' and select the desired value for beta with '--LB, 0.05'.

Example:
```
python DA_train.py  '--data-dir', 'path/to/data/folder',
                    '--num-steps', '50',
                    '--iter-size', '125'
                    '--light_discriminator', 'True',
                    '--FDA', 'True',
                    '--LB', '0.05'
                  

```
## 3. Test

The file `eval.py` contains the code to test a pretrained model.

Example:
```
python eval.py '--pretrained_model_path', 'path/to/pretrained/model',
               '--data', 'path/to/data/folder'
        
```
