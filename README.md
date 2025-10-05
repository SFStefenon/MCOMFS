#  Multi-Criteria Optimization for Model and Filter Selection

This repository presents a Multi-Criteria Optimization strategy for Model and Filter Selection (MCOMFS) applied to improve the management of hydroelectric power plants.
The strategy considers the Optuna based on a tree-structured Parzen estimator to select the predictor and the filter for the time series. The algorithm for the proposed MCOMFS is available [here](https://github.com/SFStefenon/Hydroelectric-plants/blob/main/Proposed_model.ipynb).

---

Initially, an analysis is performed to evaluate the standard model. Then the definition of the optimized model setup is handled through multi-criteria optimization, and finally, the statistical analysis and benchmarking are conducted. The proposed MCOMFS is organized as follows:

> **Loading:** Load the libraries, load the dataset, and preprocessing.

> **Standard model**: Run the TFT model considering its default setup and original dataset.

> **Filter**: Compute the evaluated filter individuality (CF, HP, STL, and MSTL).

> **Denoised model**: Run the TFT model considering the denoised time series.

> **Optuna**: Compute the multi-criteria optimization for model and filter selection.

> **Optimized**: Run the optimized model considering the denoised time series.

> **Stats**: Run several experiments and save them for statistical evaluation.

> **Benchmarking**: Compute several different models considering their default setups.

---

An example of the optimization strategy for model selection, definition of batch size, and learning rate is given:

![image](https://github.com/user-attachments/assets/59755450-118c-4099-980c-f4ef07a528cb)

The study considers the inflow data from the Belo Monte dam (the northern part of the Xingu River in the state of Pará, Brazil). For comparison purposes, daily measurements are considered for the period from December 2021 to December 2024, resulting in 1,095 observations considering three years of 365 days. The dataset used for the presented experiments is available [here](https://github.com/SFStefenon/Hydroelectric-plants/blob/main/Data/DADOS_HIDROLOGICOS_RES.csv). For analysis of other power plants' further evaluations or comparisons, the original dataset is also [available](https://github.com/SFStefenon/Hydroelectric-plants/tree/main/Data/Original).

The standard recurrent neural network (RNN), dilated RNN, long short-term memory (LSTM), temporal fusion transformer (TFT), temporal convolutional neural (TCN), deep non-parametric time series forecaster (DeepNPTS), neural basis expansion analysis for time series forecasting (N-BEATS), and neural hierarchical interpolation for time series forecasting (NHITS) models were considered. These models are available at the [Nixtla repository](https://github.com/Nixtla/neuralforecast).

The Christiano Fitzgerald (CF), Hodrick-Prescott (HP), season-trend decomposition using LOESS (STL), and multiple season-trend decomposition using LOESS (MSTL) filters are evaluated. 
These time series filters are available at the [Statsmodels website](https://www.statsmodels.org/stable/tsa.html#time-series-filters).
An example of the application of these filters is presented in the following:

![image](https://github.com/user-attachments/assets/908cbbeb-41c5-47fb-acd8-4cc28c75fd34)

---

Additional information can be found at the **[Original Paper](https://doi.org/10.1016/j.energy.2025.138688)**.

Muniz, R. N., Stefenon, S. F., Buratto, W. G., Nied, A., Cardoso, R., Yamaguchi, C. K., Yow, K. C. Time series forecasting based on multi-criteria optimization for model and filter selection applied to hydroelectric power plants, Energy, vol. 337, p. 138688, 2025.

BibTeX:
`@article{r138688, AUTHOR = {Muniz, Rafael Ninno and Stefenon, Stefano Frizzo and Buratto, William Gouv{\^e}a and Nied, Ademir and Cardoso, Rodolfo and Yamaguchi, Cristina Keiko and Yow, Kin-Choong}, TITLE = {Time series forecasting based on multi-criteria optimization for model and filter selection applied to hydroelectric power plants}, JOURNAL = {Energy}, VOLUME = {337}, YEAR = {2025}, PAGES = {138688}, DOI = {10.1016/j.energy.2025.138688}}`

---

Thank you

Dr. **Stefano Frizzo Stefenon**

Postdoctoral fellow at the University of Regina

Faculty of Engineering and Applied Sciences

Regina, SK, Canada, 2025.
