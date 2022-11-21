# 1 MAGNN
This repo is the official implementation for Multi-Scale Adaptive Graph Neural Network for Multivariate Time Series Forecasting.

## 1.1 The framework of MAGNN
 ![framework](https://github.com/shangzongjiang/MAGNN/blob/main/fig/2.png)

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

## 4.3 Hyper-parameters search with NNI
```python
# Hyper-parameters search with NNI
 nnictl create --config config.yml --port 8080
```
## 4.4 Training
```python
# Train on Solar-Energy
CUDA_LAUNCH_BLOCKING=1 python train.py --save ./model-solar-1.pt --data solar-energy/solar-energy.txt --num_nodes 137 --batch_size 4 --epochs 50 --horizon 3
# Train on Traffic
CUDA_LAUNCH_BLOCKING=1 python train.py --save ./model-traffic-2.pt --data traffic/traffic.txt --num_nodes 862 --batch_size 4 --epochs 50 --horizon 3
# Train on Electricity
CUDA_LAUNCH_BLOCKING=1 python train.py --save ./model-electricity-3.pt --data electricity/electricity.txt --num_nodes 321 --batch_size 4 --epochs 50 --horizon 3
# Train on Exchange-Rate
CUDA_LAUNCH_BLOCKING=1 python train.py --save ./model-exchange-4.pt --data exchange_rate/exchange_rate.txt --num_nodes 8 --batch_size 4 --epochs 50 --horizon 3
```
# 5 Citation
Please cite the following paper if you use the code in your work:
```
@article{chen2022multi,
  title={Multi-Scale Adaptive Graph Neural Network for Multivariate Time Series Forecasting},
  author={Chen, Ling and Chen, Donghui and Shang, Zongjiang and Zhang, Youdong and Wen, Bo and Yang, Chenghu},
  journal={arXiv preprint arXiv:2201.04828},
  year={2022}
}
```
