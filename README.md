# Deep-Learning-HRTEM-Images-Gold-Nanoparticles


This repository contains the codes to implement a Deep Learning framework to predict the 3D structure of gold nanoparticles represented in a 2D view in High-Resolution Transmission Electron Microscopy (HRTEM) images. The codes provided in this repository have been developed for a project about Computer Vision and Deep Learning applied to microscopy images of advanced materials at the University of Illinois at Chicago. The results illustrated in this repository are part of the paper "Atomic column heights detection in metallic nanoparticles using deep convolutional learning" published in the Computational Materials Science journal: 

***Ragone M.***, Yurkiv V., Song B., Ramasubramanian A., Shabazian-Yassar R., Mashayek F., Atomic column heights detection in metallic nanoparticles using deep convolutional learning, Computational Materials Science,180, (2020) 109722*


![plot](./TOC.png)

# How to use it 

## 1. Installation

**Python Installation**: 

Install Python 3.7 in Anaconda:

https://www.anaconda.com/


**Required Pyton Packages**:

**Atomic Simulation Environment, PyQSTEM, tqdm, natsort, cython**:

```yaml
conda install -c conda-forge ase pyqstem tqdm natsort cython
```

The Deep Learning models are built in both Tensorflow 2.2.0 and PyTorch. The codes are avaiolable in the folder *tf2.2* and *pytorch*. Please install the package you wish to use.

**Tensorflow 2.2.0**:

```yaml
pip3 install tensorflow==2.2.0
```
**PyTorch**:

```yaml
conda install -c pytorch pytorch torchvision
```
Both Tensorflow 2.2.0 and Pytorch contain a script for model parallelization using the library *Horovod* (https://github.com/horovod/horovod). Please refer to the linked GitHub repository for installing Horovod.

**Scikit-Lean**:

```yaml
conda install -c conda-forge scikit-learn 
```

**Scikit-Image**:

```yaml
conda install scikit-image 
```

## 2. Python codes
### 2.1 Images Simulation

The Python script to run for simulating HRTEM images and the corresponding CHs label maps are *make_NPs_data.py* or its parallelized version (much more efficient!) *make_NPs_data_multiprocessing.py*. The scripts *make_NPs_data_utils* and *make_NPs_statistics.py* are used as dependencies. The scripts generate the folders which contain the training data (simulated HRTEM image and CHs label maps).

```yaml
python make_NPs_data_multiprocessing.py
```
