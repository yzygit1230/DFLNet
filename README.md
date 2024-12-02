# DFLNet: Disentangled Feature Learning Network for Breast Cancer Ultrasound Image Segmentation
⭐ This code has been completely released ⭐ 

If our code is helpful to you, please cite:




[//]: # (* [**Requirements**]&#40;#Requirements&#41;)

[//]: # (* [**Train**]&#40;#Train&#41;)

[//]: # (* [**Test**]&#40;#Test&#41;)

[//]: # (* [**Results**]&#40;#Results&#41;)

[//]: # (* [**Time**]&#40;#Time&#41;)

[//]: # (* [**Visualization of results**]&#40;#Visualization-of-results&#41;)

[//]: # (* [**Acknowledgements**]&#40;#Acknowledgements&#41;)

[//]: # (* [**Contact**]&#40;#Contact&#41;)




## Requirements

```python
pip install -r requirements.txt
```




## Train

### 1. Prepare training data 

- The download link for the Dataset-B dataset is [here](https://pan.baidu.com/s/1iADDCBTB6r4OaxlOPRJsMQ?pwd=gmu2).
- The download link for the BUSI dataset is [here](https://pan.baidu.com/s/1Eg7pbKJVlBQ698v9B1oMsw?pwd=a9zr).
- The download link for the BUSI-WHU dataset is [here](https://pan.baidu.com/s/1Eg7pbKJVlBQ698v9B1oMsw?pwd=a9zr).
- The download link for the TN3k dataset is [here](https://pan.baidu.com/s/1Eg7pbKJVlBQ698v9B1oMsw?pwd=a9zr).
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
│   ├── valied
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

### 2. Begin to train
```python
python train.py
```


## Test

### 1. Begin to test
```python
python eval.py
```

## Quantitative Comparative Results

| Dataset       | Method (Year)     | Precision (%)    | Recall (%)       | F1 (%)           | mIOU (%)         | Kappa (%)        | ASSD            |
| ------------- | ----------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | --------------- |
| **Dataset-B** |                   |                  |                  |                  |                  |                  |                 |
|               | UNet (2015)       | 79.41            | 73.81            | 76.51            | 79.80            | 75.31            | 7.59            |
|               | MDANet (2022)     | **_blue_** 86.78 | 79.62            | 83.04            | 84.65            | 82.19            | 5.62            |
|               | MGCCNet (2023)    | 76.66            | 86.70            | 81.37            | 83.25            | 80.32            | 6.18            |
|               | EGEUNet (2023)    | 73.82            | 84.70            | 78.89            | 81.38            | 77.69            | 7.22            |
|               | DCSAUNet (2023)   | 83.84            | 84.32            | 84.08            | 85.43            | 83.24            | 5.73            |
|               | DSEUNet (2023)    | 81.14            | **_blue_** 88.95 | **_blue_** 84.87 | **_blue_** 86.02 | **_blue_** 84.03 | **_blue_** 2.34 |
|               | CDDSA (2023)      | 75.66            | **_red_** 90.07  | 82.24            | 83.89            | 81.21            | **_red_** 2.31  |
|               | SUNet (2024)      | 85.44            | 78.99            | 82.09            | 83.91            | 81.18            | 6.76            |
|               | UCTransNet (2022) | 82.39            | 83.43            | 82.91            | 84.50            | 82.00            | 2.69            |
|               | BATFormer (2023)  | 82.40            | 84.89            | 83.63            | 85.06            | 82.75            | 4.01            |
|               | LeViTUNet (2023)  | 76.87            | 80.68            | 78.89            | 82.74            | 79.61            | 3.54            |
|               | **DFLNet**        | **_red_** 91.57  | 87.25            | **_red_** 89.36  | **_red_** 89.84  | **_red_** 88.81  | 2.41            |

**Note**: Optimal values are shown in **red**, and sub-optimal values are shown in **blue**.

- Bold indicates first or second best performance.

  \begin{table*}[h]
  	\caption{Quantitative comparison results across the three datasets.  Optimal and sub-optimal values are shown in \textbf{\textcolor{red}{red}} and \textbf{\textcolor{blue}{blue}}.}
  	\centering
  	\begin{tabular}{lccccccc}
          \toprule
          Dataset & Method_{\textcolor{red}{\text{year}}} & Precision (\%) & Recall (\%) & F1 (\%) & mIOU (\%) & Kappa (\%) & ASSD \\ 
          \midrule
          \multirow{12}{*}{Dataset-B} 
          & UNet_{\text{15}} \cite{2015unet}& 79.41 & 73.81 & 76.51 & 79.80 & 75.31 & 7.59 \\ 
          & MDANet_{\text{22}} \cite{iqbal2022mda}& \textbf{\textcolor{blue}{86.78}} & 79.62 & 83.04 & 84.65 & 82.19 & 5.62 \\ 
          & MGCCNet_{\text{23}} \cite{tang2023mgcc}& 76.66 & 86.70 & 81.37 & 83.25 & 80.32 & 6.18 \\ 
          & EGEUNet_{\text{23}} \cite{ruan2023egeu}& 73.82 & 84.70 & 78.89 & 81.38 & 77.69 & 7.22 \\ 
          & DCSAUNet_{\text{23}} \cite{xu2023dcsau}& 83.84 & 84.32 & 84.08 & 85.43 & 83.24 & 5.73 \\ 
          & DSEUNet_{\text{23}} \cite{chen2023dseu}& 81.14 & \textbf{\textcolor{blue}{88.95}} & \textbf{\textcolor{blue}{84.87}} & \textbf{\textcolor{blue}{86.02}} & \textbf{\textcolor{blue}{84.03}} & \textbf{\textcolor{blue}{2.34}} \\ 
          & CDDSA_{\text{23}} \cite{gu2023cddsa}& 75.66 & \textbf{\textcolor{red}{90.07}} & 82.24 & 83.89 & 81.21 & \textbf{\textcolor{red}{2.31}} \\ 
          & SUNet_{\text{24}} \cite{ding2024sunet}& 85.44 & 78.99 & 82.09 & 83.91 & 81.18 & 6.76 \\ 
          & UCTransNet_{\text{22}} \cite{wang2022uctransnet}& 82.39 & 83.43 & 82.91 & 84.50 & 82.00 & 2.69 \\ 
          & BATFormer_{\text{23}} \cite{lin2023batformer}& 82.40 & 84.89 & 83.63 & 85.06 & 82.75 & 4.01 \\ 
          & LeViTUNet_{\text{23}} \cite{xu2023levit}& 76.87 & 80.68 & 78.89 & 82.74 & 79.61 & 3.54 \\ 
          & \textbf{DFLNet} & \textbf{\textcolor{red}{91.57}} & 87.25 & \textbf{\textcolor{red}{89.36}} & \textbf{\textcolor{red}{89.84}} & \textbf{\textcolor{red}{88.81}} & 2.41 \\ 
          \midrule
          \multirow{12}{*}{BUSI} 
          & UNet_{\text{15}} \cite{2015unet}& 75.95 & 75.96 & 75.95 & 78.25 & 73.53 & 12.48 \\ 
          & MDANet_{\text{22}} \cite{iqbal2022mda}& 78.32 & 77.06 & 77.68 & 79.58 & 75.46 & 7.66 \\ 
          & MGCCNet_{\text{23}} \cite{tang2023mgcc}& 79.28 & 81.24 & 80.24 & 81.53 & 78.23 & 5.73 \\ 
          & EGEUNet_{\text{23}} \cite{ruan2023egeu}& 68.65 & 82.33 & 74.99 & 77.28 & 72.22 & 9.37 \\ 
          & DCSAUNet_{\text{23}} \cite{xu2023dcsau}& 76.46 & 80.63 & 78.49 & 80.11 & 76.26 & 7.55 \\ 
          & DSEUNet_{\text{23}} \cite{chen2023dseu}& 75.89 & 80.75 & 78.24 & 79.91 & 75.97 & 6.99 \\ 
          & CDDSA_{\text{23}} \cite{gu2023cddsa}& \textbf{\textcolor{blue}{79.94}} & \textbf{\textcolor{blue}{84.49}} & \textbf{\textcolor{blue}{82.15}} & \textbf{\textcolor{blue}{83.03}} & \textbf{\textcolor{blue}{80.30}} & \textbf{\textcolor{red}{3.64}} \\
          & SUNet_{\text{24}} \cite{ding2024sunet}& 79.37 & 81.01 & 80.18 & 81.48 & 78.16 & \textbf{\textcolor{blue}{4.00}} \\ 
          & UCTransNet_{\text{22}} \cite{wang2022uctransnet}& 79.06 & 83.05 & 81.01 & 82.11 & 79.04 & 5.08 \\ 
          & BATFormer_{\text{23}} \cite{lin2023batformer}& 76.62 & 74.81 & 75.70 & 78.09 & 73.29 & 6.67 \\ 
          & LeViTUNet_{\text{23}} \cite{xu2023levit}& 75.32 & 77.84 & 76.56 & 78.66 & 74.15 & 8.07 \\ 
          & \textbf{DFLNet} & \textbf{\textcolor{red}{84.81}} & \textbf{\textcolor{red}{83.17}} & \textbf{\textcolor{red}{83.98}} & \textbf{\textcolor{red}{84.62}} & \textbf{\textcolor{red}{82.38}} & 5.13 \\ 
          \midrule
          \multirow{12}{*}{BUSI-WHU} 
          & UNet_{\text{15}} \cite{2015unet}& 84.07 & 85.05 & 84.56 & 85.42 & 83.34 & 3.57 \\ 
          & MDANet_{\text{22}} \cite{iqbal2022mda}& 88.03 & 87.93 & 87.98 & 88.34 & 87.04 & 1.37 \\ 
          & MGCCNet_{\text{23}} \cite{tang2023mgcc}& 89.04 & 87.34 & 88.18 & 88.52 & 87.27 & 1.57 \\ 
          & EGEUNet_{\text{23}} \cite{ruan2023egeu}& 85.65 & 86.84 & 86.24 & 86.83 & 85.15 & 2.07 \\ 
          & DCSAUNet_{\text{23}} \cite{xu2023dcsau}& 85.10 & \textbf{\textcolor{red}{89.82}} & 87.39 & 87.80 & 86.38 & 1.83 \\ 
          & DSEUNet_{\text{23}} \cite{chen2023dseu}& 89.81 & 88.32 & 89.06 & 89.30 & 88.21 & 1.27 \\ 
          & CDDSA_{\text{23}} \cite{gu2023cddsa}& 89.28 & \textbf{\textcolor{blue}{89.36}} & \textbf{\textcolor{blue}{89.32}} & \textbf{\textcolor{blue}{89.53}} & \textbf{\textcolor{blue}{88.49}} & \textbf{\textcolor{blue}{0.98}} \\ 
          & SUNet_{\text{24}} \cite{ding2024sunet}& 89.53 & 88.05 & 88.78 & 89.05 & 87.91 & 2.14 \\ 
          & UCTransNet_{\text{23}} \cite{wang2022uctransnet}& 89.75 & 86.77 & 88.23 & 88.58 & 87.33 & 1.43 \\ 
          & BATFormer_{\text{23}} \cite{lin2023batformer}& \textbf{\textcolor{blue}{90.99}} & 86.02 & 88.43 & 88.76 & 87.56 & 1.36 \\ 
          & LeViTUNet_{\text{23}} \cite{xu2023levit}& 85.82 & 89.35 & 87.55 & 87.94 & 86.56 & 1.81 \\ 
          & \textbf{DFLNet} & \textbf{\textcolor{red}{93.12}} & 89.14 & \textbf{\textcolor{red}{91.09}} & \textbf{\textcolor{red}{91.14}} & \textbf{\textcolor{red}{90.41}} & \textbf{\textcolor{red}{0.72}} \\
          \bottomrule
      \end{tabular}
      \label{tb_compare}
  \end{table*}

## Visual Comparative Results

<p align="center"> 
    <img src="Fig/Visual_B.png" width="90%"> 
</p>

<center>Visual comparison results from the Dataset-B dataset.</center>

<p align="center"> 
    <img src="Fig/Visual_BUSI.png" width="90%"> 
</p>

<center>Visual comparison results from the BUSI dataset.</center>

<p align="center"> 
    <img src="Fig/Visual_WHU.png" width="90%"> 
</p>

<center>Visual comparison results from the BUSI-WHU dataset.</center>

<p align="center"> 
    <img src="Fig/Visual_TN3k.png" width="90%"> 
</p>

<center>Visual comparison results from the TN3k dataset.</center>

