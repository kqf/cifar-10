# Transfer learning on CIFAR-10 [![Build Status](https://travis-ci.com/kqf/cifar-10.svg?token=7bkqqhrPB19pD1YKrAZM&branch=master)](https://travis-ci.com/kqf/cifar-10)
CIFAR-10 dataset is a collection of 10 classes of 32x32 images
![cifar-10](exploration/cifar10.png)

This dataset can be visualized as two-dimensional embedding:
![pixels](exploration/pixel-features.png)


## Solution
The main idea is to use large CNN network as encoder to extract features from these images automatically, and train a model on top of these features. It looks like ResNet50 architecture is a good choice due to low memory and CPU cost.

![ResNet](exploration/ResNet50.png)

Two-dimensional representation of ResNet-50 features using tSNE:
![ResNet-features](exploration/resnet50-features.png)

The parameters are still to be tuned, as this task is memory intensive.


## Install
```bash

# First install cython
pip install cython

# Install other requirements
pip install -r requirements.txt

# Then install the package iteself
pip install .
```


## Run the solution
To run the solution do
```bash
make 
```
