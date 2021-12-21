# 1 MAGNN
This repo is the official implementation for Multi-Scale Adaptive Graph Neural Network for Multivariate Time Series Forecasting.

## 1.1 The framework of SNAS4MTF
 ![framework](https://user-images.githubusercontent.com/18440709/142983473-f6bb12bd-266a-40ce-9116-00ce9a9f1059.jpg)

# 2 Prerequisites

* Python 3.6.12
* PyTorch 1.2.0
* math, sklearn, numpy
# 3 Datasets
## 3.1 METR-LA
This dataset is collected by the Los Angeles Metropolitan Transportation Authority and contains the average traffic speed measured by 207 loop detectors on the highways of Los Angeles County between March 2012 and June 2012.
## 3.2 PEMS-BAY
The raw data is in http://pems.dot.ca.gov. This dataset is collected by California Transportation Agencies and contains the average traffic speed measured by 325 sensors in the Bay Area between January 2017 and May 2017.
# 4 Running
## 4.1 Install all dependencies listed in prerequisites

## 4.2 Download the dataset

## 4.3 Neural Architecture Search
```python
# Neural Architecture Search on PEMS_BAY
 python search.py --config config/PEMS_BAY_para.yaml |& tee logs/search_PEMS_BAY.log
 # Neural Architecture Search on METR_LA
 python search.py --config config/METR_LA_para.yaml |& tee logs/search_METR_LA.log
```
## 4.4 Training
```python
# Train on PEMS_BAY
python train.py --config config/PEMS_BAY_para.yaml  |& tee logs/train_PEMS_BAY.log
# Train on METR-LA
python train.py --config config/METR_LA_para.yaml |& tee logs/train_METR_LA.log
```
## 4.5 Evaluating
```python
# Evaluate on PEMS_BAY
python test.py --config config/PEMS_BAY_para.yaml |& tee logs/test_PEMS_BAY.log
# Evaluate on METR-LA
python test.py --config config/METR_LA_para.yaml |& tee logs/test_METR_LA.log
```
# 5 Citation
Please cite the following paper if you use the code in your work:
```
@Inproceedings{616B,
  title={Scale-Aware Neural Architecture Search for Multivariate Time Series Forecasting.},
  author={Donghui Chen, Ling Chen, Youdong Zhang, et al.},
  booktitle={},
  year={2021}
}
```
