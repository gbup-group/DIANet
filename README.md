# DIANet: Dense-and-Implicit-Attention-Network
[![996.ICU](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu) 
![GitHub](https://img.shields.io/github/license/gbup-group/DIANet.svg)
![GitHub](https://img.shields.io/badge/gbup-%E7%A8%B3%E4%BD%8F-blue.svg)

By [Zhongzhan Huang](https://github.com/dedekinds), [Senwei Liang](https://github.com/LeungSamWai), [Mingfu Liang](https://github.com/wuyujack) and [Haizhao Yang](https://haizhaoyang.github.io/).

This repo is the official implementation of "DIANet: Dense-and-Implicit Attention Network" [[paper]](www.baidu.com)  on CIFAR-100 dataset.

## Introduction

DIANet[[paper]](www.google.com) provides a universal framework that recurrently fuses the information from preceding layers to enhance the attention modeling at each layer. The existing network backbone can embed DIA unit readily by sharing the DIA unit with all the layers. In the implementation, an LSTM-based DIA unit is provided.



![image](https://github.com/gbup-group/DIANet/blob/master/image/fig4.jpg)


## Install
* Install [PyTorch](http://pytorch.org/)
* Clone the DIANet repository
  ```
  git clone https://github.com/gbup-group/DIANet.git
  ```

## Usage
  ```
 python run_code.py
  ```



## Results
|                 | original | DIANet(r=4) |
|:---------------:|:--------:|:------:|
|    ResNet164    |   73.43  |  76.67 |
|   PreResNet164  |   76.53  |  78.20 |
|     WRN52-4     |   79.75  |  80.99 |
| ResNext101,8x32 |   81.18  |  82.46 |


**Notes:**

- After five times experiments, we show the mean performance in the table above. 
- `r` denote reduction ratio in DIA unit. 
- For more implementation details and parameter settings check the code or the appendix of our paper.


## Citing DIANet

```
xxxxxhi
```
## Acknowledgments
Many thanks to [bearpaw](https://github.com/bearpaw) for his simple and clean [framework](https://github.com/bearpaw/pytorch-classification) for image classification task(PyTorch).
