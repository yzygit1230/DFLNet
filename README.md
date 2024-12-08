# DFLNet: Disentangled Feature Learning Network for Breast Cancer Ultrasound Image Segmentation
⭐ This code has been completely released ⭐ 

## Overview

<p align="center"> 
    <img src="Fig/DFLNet.png" width="85%"> 
</p>

## Requirements

```python
Python 3.6
Pytorch 1.7.0
```


## Datasets Preparation

- The download link for the datasets is [here](https://pan.baidu.com/s/1vaLegDjMUvQjaNA4qNsBzw?pwd=fqsu). Put the datasets as follows:
```python
DFLNet
├── BUSI-WHU
│   ├── train
│   │   ├── img
│   │   │   ├── 00001.bmp
│   │   │   ├── 00002.bmp
│   │   │   ├── .....
│   │   ├── gt
│   │   │   ├── 00001.bmp
│   │   │   ├── 00002.bmp
│   │   │   ├── .....
│   ├── valid
│   │   ├── img
│   │   │   ├── 00009.bmp
│   │   │   ├── 00015.bmp
│   │   │   ├── .....
│   │   ├── gt
│   │   │   ├── 00009.bmp
│   │   │   ├── 00015.bmp
│   │   │   ├── .....
│   ├── test
│   │   ├── img
│   │   │   ├── 00007.bmp
│   │   │   ├── 00008.bmp
│   │   │   ├── .....
│   │   ├── gt
│   │   │   ├── 00007.bmp
│   │   │   ├── 00008.bmp
│   │   │   ├── .....
```

## Train

Modify the paths in lines 22 to 30 of the train.py, then simply run:

```python
python train.py
```

- The download link for the pretrain_pth is [here](https://pan.baidu.com/s/1Yn5vZEPhtJXE57x3jVV21Q?pwd=rdeg).

## Test

Modify the paths in lines 14 to 15 of the eval.py, then simply run:

```python
python eval.py
```

## Visualization

Modify the paths in lines 13 to 17 of the visualization.py, then simply run:

```
python visualization.py
```

- Note that batch-size must be 1 when using visualization.py
- Besides, you can adjust the parameter of full_to_color to change the color

## Visual Results

<p align="center"> 
    <img src="Fig/Visual.png" width="90%"> 
</p>
