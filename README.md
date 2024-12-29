# Hydroelectric-plants

This repository presents a multi-criteria optimization strategy for model and filter selection applied to improve the management of hydroelectric power plants.
The strategy considers the Optuna based on a tree-structured Parzen estimator to select the predictor and the filter for the time series. The algorithm for the proposed method is available here.

![image](https://github.com/user-attachments/assets/3b62062c-a716-49be-92e1-d0f7323e1b05)



The study considers the inflow data from the Belo Monte dam in Brazil. For comparison purposes, daily measurements are considered for the period from December 2021 to December 2024, resulting in 1,095 observations considering three years of 365 days. The dataset is available [here](), for analysis of other power plants' further evaluations or comparisons can be made based on the [original data](https://dados.ons.org.br/dataset/dados-hidrologicos-res).

The standard recurrent neural network (RNN), dilated RNN, long short-term memory (LSTM), temporal fusion transformer (TFT), temporal convolutional neural (TCN), deep non-parametric time series forecaster (DeepNPTS), neural basis expansion analysis for time series forecasting (N-BEATS), and neural hierarchical interpolation for time series forecasting (NHITS) models were considered.

The Christiano Fitzgerald (CF), Hodrick-Prescott (HP), season-trend decomposition using LOESS (STL), and multiple season-trend decomposition using LOESS (MSTL) filters are evaluated. An example of the application of these filters is presented in the following:

![image](https://github.com/user-attachments/assets/908cbbeb-41c5-47fb-acd8-4cc28c75fd34)


Thank you.

Dr. Stefano Frizzo Stefenon 

Regina, Canada, January 01, 2025.











