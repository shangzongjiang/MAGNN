# 1 MAGNN
This repo is the official implementation for Multi-Scale Adaptive Graph Neural Network for Multivariate Time Series Forecasting.

## 1.1 The framework of MAGNN
 ![framework](https://github.com/shangzongjiang/MAGNN/blob/main/fig/2.pdf)

# 2 Prerequisites

* Python 3.6.12
* PyTorch 1.0.0
* math, sklearn, numpy
# 3 Datasets
To evaluate the performance of MAGNN, we conduct experiments on [four public benchmark datasets](https://github.com/laiguokun/multivariate-time-series-data)：Solar-Energy, Traffic, Electricity, and Exchange-Rate.
## 3.1 Solar-Energy
This dataset contains the collected solar power from the National Renewable Energy Laboratory, which is sampled every 10 minutes from 137 PV plants in Alabama State in 2007.
## 3.2 Traffic
This dataset contains the road occupancy rates (between 0 and 1) from the California Department of Transportation, which is hourly aggregated from 862 sensors in San Francisco Bay Area from 2015 to 2016.
## 3.3 Electricity
This dataset contains the electricity consumption from the UCI Machine Learning Repository, which is hourly aggregated from 321 clients from 2012 to 2014.
## 3.4 Exchange-Rate
This dataset contains the exchange rates of eight countries, which is sampled daily from 1990 to 2016.
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
