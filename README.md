#  Multi-criteria optimization for model and filter selection

This repository presents a multi-criteria optimization strategy for model and filter selection (MCOMFS) applied to improve the management of hydroelectric power plants.
The strategy considers the Optuna based on a tree-structured Parzen estimator to select the predictor and the filter for the time series. The algorithm for the proposed method is available [here](https://github.com/SFStefenon/Hydroelectric-plants/blob/main/Proposed_model.ipynb).

The study considers the inflow data from the Belo Monte dam in Brazil. For comparison purposes, daily measurements are considered for the period from December 2021 to December 2024, resulting in 1,095 observations considering three years of 365 days. The considered dataset is available [here](https://github.com/SFStefenon/Hydroelectric-plants/blob/main/Data/DADOS_HIDROLOGICOS_RES.csv). For analysis of other power plants' further evaluations or comparisons, the original dataset is available [here](https://github.com/SFStefenon/Hydroelectric-plants/tree/main/Data/Original).

The standard recurrent neural network (RNN), dilated RNN, long short-term memory (LSTM), temporal fusion transformer (TFT), temporal convolutional neural (TCN), deep non-parametric time series forecaster (DeepNPTS), neural basis expansion analysis for time series forecasting (N-BEATS), and neural hierarchical interpolation for time series forecasting (NHITS) models were considered.

The Christiano Fitzgerald (CF), Hodrick-Prescott (HP), season-trend decomposition using LOESS (STL), and multiple season-trend decomposition using LOESS (MSTL) filters are evaluated. An example of the application of these filters is presented in the following:

![image](https://github.com/user-attachments/assets/908cbbeb-41c5-47fb-acd8-4cc28c75fd34)


Thank you;

Dr. **Stefano Frizzo Stefenon**

Postdoctoral fellow at the University of Regina

Faculty of Engineering and Applied Sciences

Regina, SK, Canada, 2025.
