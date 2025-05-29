"""
Wrote by Dr. Stefano Frizzo Stefenon
Postdoctoral fellow at the University of Regina
Faculty of Engineering and Applied Sciences
Regina, SK, Canada, 2025.
___
After loading the libraries and the dataset, the proposed analysis is divided according to:
*   Standard model: Run the TFT model considering its default setup and original dataset.
*   Filter: Compute the evaluated filter individuality (CF, HP, STL, and MSTL).
*   Denoised model: Run the TFT model considering the denoised time series.
*   Optuna: Compute the multi-criteria optimization for model and filter selection.
*   Optimized: Run the optimized model considering the denoised time series.
*   Stats: Run several experiments and save them for statistical evaluation.
*   Benchmarking: Compute several different models considering their default setups.
"""

# pip install -q neuralforecast

from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.models import DilatedRNN
from neuralforecast.models import NHITS
from neuralforecast.models import DeepNPTS
from neuralforecast.models import TCN
from neuralforecast.models import LSTM
from neuralforecast.models import RNN
from neuralforecast.models import MLP
from neuralforecast.models import NBEATS

import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import MSTL
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error

def smape(a, f): # Compute SMAPE
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)
def make_Tensor(array): # Tensor transformation
    return torch.from_numpy(array).float()

"""
## Data - Original
"""

# Load the dataset
data_o = pd.read_csv("DADOS_HIDROLOGICOS_RES.csv", delimiter=';', decimal='.')
inflow_o = list(data_o['val_vazaoafluente'])
time = list(data_o['din_instante'])
years = 3
inflow = inflow_o[(-365*years):-1]

fig, ((ax2)) = plt.subplots(1, 1, figsize=(5, 3), dpi=100)
ax2.plot(inflow,'k', zorder=2)
ax2.set_ylabel('Flow (m$^3$/s)')
ax2.set_xlabel('Time (days)')
ax2.axis([0, len(inflow), 0, 40000])
ax2.grid(linestyle = '--', linewidth = 0.5, zorder=0)
plt.show
plt.savefig('Original.pdf', bbox_inches = 'tight')

"""
# Standard Model
"""

# Load the dataset
data_o = pd.read_csv("DADOS_HIDROLOGICOS_RES.csv", delimiter=';', decimal='.')
inflow_o = list(data_o['val_vazaoafluente'])
time = list(data_o['din_instante'])
inflow = inflow_o[(-365*3):-1]

# Create the time data
horizon = 14
x = inflow
time_data = pd.date_range(time[0], periods=len(x), freq='D')

# Create the DataFrame
df = pd.DataFrame({
    'unique_id': np.array(['Airline1'] * len(x)),
    'ds': time_data,
    'y': x,
    'trend': np.arange(len(x)),})

# Display the DataFrame
print(df.head())
print(df.tail())

# Split data based on your criteria
# Calculate split point for 70-10 split
split_point = int(len(df) * 0.70)

# Split data into 90% training and 10% testing
Y_train_df = df.iloc[:split_point].reset_index(drop=True)
Y_test_df = df.iloc[split_point:].reset_index(drop=True)

# Adjust input_size to fit within the training data length
input_size = min(2 * horizon, len(Y_train_df))

import time
start = time.time()
models = [TFT(input_size=horizon, h=horizon, max_steps=100)]
nf = NeuralForecast(models=models, freq='D')
nf.fit(df=Y_train_df)
Y_hat_df = nf.predict().reset_index()
end = time.time()
time_s = end - start #save time

y_pred = Y_hat_df.TFT[:horizon]
y_true = Y_test_df.y[:horizon]
rmse(y_true, y_pred)

# RMSE & MAE & MAPE  & SMAPE
print(f'{(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(smape(y_true, y_pred)):.2E} & {time_s:.2f}')

"""
# Filter
"""

x = inflow
f = pd.Series(x, index=pd.date_range(x[0], periods=len(x), freq="D"), name="DF")

# Denoising
x_resd_cf, x_trend_cf = sm.tsa.filters.cffilter(f, 2, 35, False)
x_cf = list(x_trend_cf)
x_resd_hp, x_trend_hp = sm.tsa.filters.hpfilter(f, 35)
x_hp = list(x_trend_hp)
x_trend_mstl = MSTL(f, periods=(24, 35)).fit()
x_mstl = list(x_trend_mstl.trend)
stl = STL(f, trend=35, seasonal=7)
x_stl = list((stl.fit()).trend)

fig, ((ax1)) = plt.subplots(1, figsize=(7, 3), dpi=100)

ax1.plot(x,'k', zorder=0)
ax1.plot(x_cf,'b', zorder=2, linestyle='dashed')
ax1.plot(x_hp,'r', zorder=2, linestyle='dashed')
ax1.plot(x_stl,'m', zorder=2, linestyle='dashed')
ax1.plot(x_mstl,'g', zorder=2, linestyle='dashed')
ax1.set_ylabel('Flow (m$^3$/s)')
ax1.set_xlabel('Time (days)')
ax1.set_xlim(0, 220)
ax1.set_ylim(0, 40000)
ax1.grid(linestyle = '--', linewidth = 0.5, zorder=0)
ax1.legend(["Original", "CF filter", "HP filter", "MSTL filter", "STL filter"], loc="upper right", fancybox=True, shadow=False, ncol=1)
plt.show
plt.savefig('Trend.pdf', bbox_inches = 'tight')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8), dpi=100)

ax1.plot(x,'k', zorder=2)
ax1.plot(x_cf,'r', zorder=2)
ax1.set_title('A)', loc='left')
ax1.set_ylabel('Flow (m$^3$/s)')
ax1.set_xlabel('Time (days)')
ax1.set_xlim(0, len(x))
ax1.set_ylim(0, 40000)
ax1.grid(linestyle = '--', linewidth = 0.5, zorder=0)
ax1.legend(["Original", "CF filter"], loc="upper right", fancybox=True, shadow=False, ncol=1)

ax2.plot(x,'k', zorder=2)
ax2.plot(x_hp,'b', zorder=2)
ax2.set_title('B)', loc='left')
ax2.set_ylabel('Flow (m$^3$/s)')
ax2.set_xlabel('Time (days)')
ax2.set_xlim(0, len(x))
ax2.set_ylim(0, 40000)
ax2.grid(linestyle = '--', linewidth = 0.5, zorder=0)
ax2.legend(["Original", "HP filter"], loc="upper right", fancybox=True, shadow=False, ncol=1)

ax3.plot(x,'k', zorder=2)
ax3.plot(x_stl,'m', zorder=2)
ax3.set_title('C)', loc='left')
ax3.set_ylabel('Flow (m$^3$/s)')
ax3.set_xlabel('Time (days)')
ax3.set_xlim(0, len(x))
ax3.set_ylim(0, 40000)
ax3.grid(linestyle = '--', linewidth = 0.5, zorder=0)
ax3.legend(["Original", "STL filter"], loc="upper right", fancybox=True, shadow=False, ncol=1)

ax4.plot(x,'k', zorder=2)
ax4.plot(x_mstl,'g', zorder=2)
ax4.set_title('D)', loc='left')
ax4.set_ylabel('Flow (m$^3$/s)')
ax4.set_xlabel('Time (days)')
ax4.set_xlim(0, len(x))
ax4.set_ylim(0, 40000)
ax4.grid(linestyle = '--', linewidth = 0.5, zorder=0)
ax4.legend(["Original", "MSTL filter"], loc="upper right", fancybox=True, shadow=False, ncol=1)
plt.show
plt.savefig('Trend.pdf', bbox_inches = 'tight')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8), dpi=100)

ax1.plot(x,'k', zorder=2)
ax1.plot(x_cf,'r', zorder=2)
ax1.set_title('A)', loc='left')
ax1.set_ylabel('Flow (m$^3$/s)')
ax1.set_xlabel('Time (days)')
ax1.set_xlim(0, len(x)*.2)
ax1.set_ylim(0, 40000)
ax1.grid(linestyle = '--', linewidth = 0.5, zorder=0)
ax1.legend(["Original", "CF filter"], loc="upper right", fancybox=True, shadow=False, ncol=1)

ax2.plot(x,'k', zorder=2)
ax2.plot(x_hp,'b', zorder=2)
ax2.set_title('B)', loc='left')
ax2.set_ylabel('Flow (m$^3$/s)')
ax2.set_xlabel('Time (days)')
ax2.set_xlim(0, len(x)*.2)
ax2.set_ylim(0, 40000)
ax2.grid(linestyle = '--', linewidth = 0.5, zorder=0)
ax2.legend(["Original", "HP filter"], loc="upper right", fancybox=True, shadow=False, ncol=1)

ax3.plot(x,'k', zorder=2)
ax3.plot(x_stl,'m', zorder=2)
ax3.set_title('C)', loc='left')
ax3.set_ylabel('Flow (m$^3$/s)')
ax3.set_xlabel('Time (days)')
ax3.set_xlim(0, len(x)*.2)
ax3.set_ylim(0, 40000)
ax3.grid(linestyle = '--', linewidth = 0.5, zorder=0)
ax3.legend(["Original", "STL filter"], loc="upper right", fancybox=True, shadow=False, ncol=1)

ax4.plot(x,'k', zorder=2)
ax4.plot(x_mstl,'g', zorder=2)
ax4.set_title('D)', loc='left')
ax4.set_ylabel('Flow (m$^3$/s)')
ax4.set_xlabel('Time (days)')
ax4.set_xlim(0, len(x)*.2)
ax4.set_ylim(0, 40000)
ax4.grid(linestyle = '--', linewidth = 0.5, zorder=0)
ax4.legend(["Original", "MSTL filter"], loc="upper right", fancybox=True, shadow=False, ncol=1)
plt.show
plt.savefig('Trend.pdf', bbox_inches = 'tight')

"""# Denoised model"""

# Load the dataset
data_o = pd.read_csv("DADOS_HIDROLOGICOS_RES.csv", delimiter=';', decimal='.')
inflow_o = list(data_o['val_vazaoafluente'])
time = list(data_o['din_instante'])
x = inflow_o[(-365*3):-1]
f = pd.Series(x, index=pd.date_range(x[0], periods=len(x), freq="D"), name="DF")

# Denoising
x_resd_cf, x_trend_cf = sm.tsa.filters.cffilter(f, 2, 35, False)
x_cf = list(x_trend_cf)
x_resd_hp, x_trend_hp = sm.tsa.filters.hpfilter(f, 35)
x_hp = list(x_trend_hp)
x_trend_mtsl = MSTL(f, periods=(24, 35)).fit()
x_mstl = list(x_trend_mtsl.trend)
stl = STL(f, trend=35, seasonal=7)
x_stl = list((stl.fit()).trend)

# x = x_cf
# x = x_hp
x = x_mstl
# x = x_stl

# Create the time data
horizon = 14
time_data = pd.date_range(time[0], periods=len(x), freq='D')

# Create the DataFrame
df = pd.DataFrame({
    'unique_id': np.array(['Airline1'] * len(x)),
    'ds': time_data,
    'y': x,
    'trend': np.arange(len(x)),})

# Split data based on your criteria
split_point = int(len(df) * 0.7)
Y_train_df = df.iloc[:split_point].reset_index(drop=True)
Y_test_df = df.iloc[split_point:].reset_index(drop=True)

# Adjust input_size to fit within the training data length
input_size = min(2 * horizon, len(Y_train_df))

import time
start = time.time()
models = [TFT(input_size=horizon, h=horizon, max_steps=100)]
nf = NeuralForecast(models=models, freq='D')
nf.fit(df=Y_train_df)
Y_hat_df = nf.predict().reset_index()
end = time.time()
time_s = end - start #save time

y_pred = Y_hat_df.TFT[:horizon]
y_true = Y_test_df.y[:horizon]
rmse(y_true, y_pred)

# RMSE & MAE & MAPE  & SMAPE
print(f'{(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(smape(y_true, y_pred)):.2E} & {time_s:.2f}')

"""# Optuna

"""

!pip install -q optuna
import optuna

# Load the dataset
horizon = 14
data_o = pd.read_csv("DADOS_HIDROLOGICOS_RES.csv", delimiter=';', decimal='.')
inflow_o = list(data_o['val_vazaoafluente'])
time = list(data_o['din_instante'])

"""## Exp 1

"""

def attention(a,b):
  x = inflow_o[(-365*3):-1]
  f = pd.Series(x, index=pd.date_range(x[0], periods=len(x), freq="D"), name="DF")
  if b == "Original":
    pass
  elif b == "CF":
    x_resd_cf, x_trend_cf = sm.tsa.filters.cffilter(f, 2, a, False)
    x = list(x_trend_cf)
  elif b == "HP":
    x_resd_hp, x_trend_hp = sm.tsa.filters.hpfilter(f, a)
    x = list(x_trend_hp)
  elif b == "STL":
    if a % 2 == 0:
      a = a+1 # To keep only odds for the STL
    stl = STL(f, seasonal=7, trend=a)
    x_stl = list((stl.fit()).trend)
  elif b == "MSTL":
    x_trend_mstl = MSTL(f, periods=(24, a)).fit()
    x_mstl = list(x_trend_mstl.trend)

  time_data = pd.date_range(time[0], periods=len(x), freq='D')
  df = pd.DataFrame({'unique_id': np.array(['Airline1'] * len(x)),
      'ds': time_data, 'y': x, 'trend': np.arange(len(x)),})
  split_point = int(len(df) * 0.70)
  Y_train_df = df.iloc[:split_point].reset_index(drop=True)
  Y_test_df = df.iloc[split_point:].reset_index(drop=True)
  input_size = min(2 * horizon, len(Y_train_df))

  models = [TFT(input_size=horizon, h=horizon, max_steps=30)]
  nf = NeuralForecast(models=models, freq='d')
  nf.fit(df=Y_train_df)
  Y_hat_df = nf.predict().reset_index()
  y_pred = Y_hat_df.TFT[:horizon]
  y_true = Y_test_df.y[:horizon]
  rmse(y_true, y_pred)
  return rmse(y_true, y_pred)

def objective(trial):
    a = trial.suggest_int('Hyperparameter', 1, 100)
    b = trial.suggest_categorical("Input_data", ["Original", "CF", "HP", "STL", "MSTL"])
    erro = attention(a,b)
    return erro

study = optuna.create_study()
study.optimize(objective, n_trials=50)

study.best_params

print("Best params: ", study.best_params)

optuna.visualization.plot_param_importances(study)

optuna.visualization.plot_contour(study, params=["Hyperparameter", "Input_data"])

optuna.visualization.matplotlib.plot_parallel_coordinate(study, params=["Hyperparameter", "Input_data"])

"""## Exp 2"""

max_steps = 30
n_trials = 50

def attention(a,b,c):
  x = inflow_o[(-365*3):-1]
  f = pd.Series(x, index=pd.date_range(x[0], periods=len(x), freq="D"), name="DF")
  x_resd_hp, x_trend_hp = sm.tsa.filters.hpfilter(f, 24)
  x = list(x_trend_hp)
  time_data = pd.date_range(time[0], periods=len(x), freq='D')
  df = pd.DataFrame({'unique_id': np.array(['Airline1'] * len(x)),
      'ds': time_data, 'y': x, 'trend': np.arange(len(x)),})
  split_point = int(len(df) * 0.70)
  Y_train_df = df.iloc[:split_point].reset_index(drop=True)
  Y_test_df = df.iloc[split_point:].reset_index(drop=True)
  input_size = min(2 * horizon, len(Y_train_df))
  RMSE = 1000

  if b == "StandardRNN":
    models = [RNN (input_size=horizon, h=horizon, learning_rate=a, batch_size=c, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq='d')
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    y_pred = Y_hat_df.RNN[:horizon]
    y_true = Y_test_df.y[:horizon]
    RMSE = rmse(y_true, y_pred)
  elif b == "DilatedRNN":
    models = [DilatedRNN (input_size=horizon, h=horizon, learning_rate=a, batch_size=c, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq='d')
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    y_pred = Y_hat_df.DilatedRNN [:horizon]
    y_true = Y_test_df.y[:horizon]
    RMSE = rmse(y_true, y_pred)
  elif b == "LSTM":
    models = [LSTM (input_size=horizon, h=horizon, learning_rate=a, batch_size=c, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq='d')
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    y_pred = Y_hat_df.LSTM[:horizon]
    y_true = Y_test_df.y[:horizon]
    RMSE = rmse(y_true, y_pred)
  elif b == "TFT":
    models = [TFT(input_size=horizon, h=horizon, learning_rate=a, batch_size=c, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq='d')
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    y_pred = Y_hat_df.TFT[:horizon]
    y_true = Y_test_df.y[:horizon]
    RMSE = rmse(y_true, y_pred)
  elif b == "NHITS":
    models = [NHITS(input_size=horizon, h=horizon, learning_rate=a, batch_size=c, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq='d')
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    y_pred = Y_hat_df.NHITS[:horizon]
    y_true = Y_test_df.y[:horizon]
    RMSE = rmse(y_true, y_pred)
  elif b == "DeepNPTS":
    models = [DeepNPTS(input_size=horizon, h=horizon, learning_rate=a, batch_size=c, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq='d')
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    y_pred = Y_hat_df.DeepNPTS[:horizon]
    y_true = Y_test_df.y[:horizon]
    RMSE = rmse(y_true, y_pred)
  elif b == "NBEATS":
    models = [NBEATS(input_size=horizon, h=horizon, learning_rate=a, batch_size=c, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq='d')
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    y_pred = Y_hat_df.NBEATS[:horizon]
    y_true = Y_test_df.y[:horizon]
    RMSE = rmse(y_true, y_pred)
  elif b == "TCN":
    models = [TCN(input_size=horizon, h=horizon, learning_rate=a, batch_size=c, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq='d')
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    y_pred = Y_hat_df.TCN[:horizon]
    y_true = Y_test_df.y[:horizon]
    RMSE = rmse(y_true, y_pred)
  elif b == "DeepNPTS":
    models = [DeepNPTS(input_size=horizon, h=horizon, learning_rate=a, batch_size=c, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq='d')
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    y_pred = Y_hat_df.DeepNPTS[:horizon]
    y_true = Y_test_df.y[:horizon]
    RMSE = rmse(y_true, y_pred)
  elif b == "NBEATS":
    models = [NBEATS(input_size=horizon, h=horizon, learning_rate=a, batch_size=c, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq='d')
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    y_pred = Y_hat_df.NBEATS[:horizon]
    y_true = Y_test_df.y[:horizon]
    RMSE = rmse(y_true, y_pred)
  elif b == "NHITS":
    models = [NHITS(input_size=horizon, h=horizon, learning_rate=a, batch_size=c, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq='d')
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    y_pred = Y_hat_df.NHITS[:horizon]
    y_true = Y_test_df.y[:horizon]
    RMSE = rmse(y_true, y_pred)

  #print(f'y_true: {y_true}, y_pred: {y_pred}')
  return RMSE

def objective(trial):
    a = trial.suggest_float('learning_rate', 0.001, 0.01)
    b = trial.suggest_categorical("model", ["Standard RNN", "Dilated RNN", "LSTM", "TFT", "TCN", "DeepNPTS", "N-BEATS", "NHITS"])
    c = trial.suggest_int('batch_size', 2, 32)
    erro = attention(a,b,c)
    return erro

study = optuna.create_study()
study.optimize(objective, n_trials=50)
study.best_params

print("Best params: ", study.best_params)

optuna.visualization.plot_param_importances(study)

optuna.visualization.plot_contour(study, params = ["learning_rate", "model", "batch_size"])

optuna.visualization.matplotlib.plot_parallel_coordinate(study, params = ["learning_rate", "model", "batch_size"])

"""# Optimized"""

# Load the dataset
data_o = pd.read_csv("DADOS_HIDROLOGICOS_RES.csv", delimiter=';', decimal='.')
inflow_o = list(data_o['val_vazaoafluente'])
time = list(data_o['din_instante'])
inflow = inflow_o[(-365*3):-1]

# Create the time data
horizon = 14
x = inflow
time_data = pd.date_range(time[0], periods=len(x), freq='D')

# Create the DataFrame
df = pd.DataFrame({
    'unique_id': np.array(['Airline1'] * len(x)),
    'ds': time_data,
    'y': x,
    'trend': np.arange(len(x)),})

# Display the DataFrame
print(df.head())
print(df.tail())

# Split data based on your criteria
# Calculate split point for 70-10 split
split_point = int(len(df) * 0.70)

# Split data into 90% training and 10% testing
Y_train_df = df.iloc[:split_point].reset_index(drop=True)
Y_test_df = df.iloc[split_point:].reset_index(drop=True)

# Adjust input_size to fit within the training data length
input_size = min(2 * horizon, len(Y_train_df))

import time
start = time.time()

models = [TFT(input_size=horizon, h=horizon, max_steps=100, learning_rate=0.00303374209742937, batch_size= 9)]
nf = NeuralForecast(models=models, freq='d')
nf.fit(df=Y_train_df)
Y_hat_df = nf.predict().reset_index()

end = time.time()
time_s = end - start #save time

y_pred = Y_hat_df.TFT[:horizon]
y_true = Y_test_df.y[:horizon]

# RMSE & MAE & MAPE  & SMAPE
print(f'{(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(smape(y_true, y_pred)):.2E} & {time_s:.2f}')

fig, ((ax1)) = plt.subplots(1, figsize=(5, 3), dpi=100)

ax1.plot([0]+list(y_true),'k', zorder=2)
ax1.plot([0]+list(y_pred),'-r', zorder=2)
ax1.set_ylabel('Flow (m$^3$/s)')
ax1.set_xlabel('Time (days)')
ax1.set_xlim(1, 14)
ax1.set_ylim(2000, 10000)
ax1.grid(linestyle = '--', linewidth = 0.5, zorder=0)
ax1.legend(["Observed", "Predicted"], loc="upper right", fancybox=True, shadow=False, ncol=1)
plt.show
plt.savefig('comp.pdf', bbox_inches = 'tight')

"""# Stats"""

horizon = 14

erro=[]
for k in range(0,50):
  x = inflow_o[(-365*3):-1]
  f = pd.Series(x, index=pd.date_range(x[0], periods=len(x), freq="D"), name="DF")
  x_resd_hp, x_trend_hp = sm.tsa.filters.hpfilter(f, 24)
  x = list(x_trend_hp)
  time_data = pd.date_range(time[0], periods=len(x), freq='D')
  df = pd.DataFrame({'unique_id': np.array(['Airline1'] * len(x)),
      'ds': time_data, 'y': x, 'trend': np.arange(len(x)),})
  split_point = int(len(df) * 0.70)
  Y_train_df = df.iloc[:split_point].reset_index(drop=True)
  Y_test_df = df.iloc[split_point:].reset_index(drop=True)
  input_size = min(2 * horizon, len(Y_train_df))

  print(k, 'iterations')
  models = [TFT(input_size=horizon, h=horizon, max_steps=100, random_seed=k, learning_rate=0.00303374209742937, batch_size= 9)]
  nf = NeuralForecast(models=models, freq='h')
  nf.fit(df=Y_train_df)
  Y_hat_df = nf.predict().reset_index()
  y_pred = Y_hat_df.TFT[:horizon]
  y_true = Y_test_df.y[:horizon]
  r = (rmse(y_true, y_pred)), (mean_absolute_error(y_true, y_pred)), (mean_absolute_percentage_error(y_true, y_pred)), (smape(y_true, y_pred))
  erro.append(r)

erro = np.array(erro)
# Creating the dataframe
dfphi = pd.DataFrame({'rmse': erro[:, 0], 'mae': erro[:, 1], 'mape': erro[:, 2], 'smape': erro[:, 3]})
pd.DataFrame(erro[:, 0]).to_csv("rmse.csv", index=True)
pd.DataFrame(erro[:, 1]).to_csv("mae.csv", index=True)
pd.DataFrame(erro[:, 2]).to_csv("mape.csv", index=True)
pd.DataFrame(erro[:, 3]).to_csv("smape.csv", index=True)

# calculate statistics
summary_df = pd.DataFrame({
    'Mean': dfphi.mean(),
    'Median': dfphi.median(),
    'Mode': dfphi.mode().iloc[0],
    'Range': dfphi.max() - dfphi.min(),
    'Variance': dfphi.var(),
    'Std. Dev.': dfphi.std(),
    '25th %ile': dfphi.quantile(0.25),
    '50th %ile': dfphi.quantile(0.50),
    '75th %ile': dfphi.quantile(0.75),
    'IQR': dfphi.quantile(0.75) - dfphi.quantile(0.25),
    'Skewness': dfphi.skew(),
    'Kurtosis': dfphi.kurtosis()
})

# print summary to latex
print(summary_df.transpose().round(5).to_latex())
plt.figure(figsize=(7, 3))
plt.boxplot(dfphi)
plt.legend()
plt.grid(linestyle='-', which='both')
plt.show()
plt.savefig("box.pdf", dpi=400, bbox_inches = 'tight')

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d1 = pd.read_csv("rmse.csv", index_col=0)
d2 = pd.read_csv("mae.csv", index_col=0)
d3 = pd.read_csv("mape.csv", index_col=0)
d4 = pd.read_csv("smape.csv", index_col=0)

# Combine the data into a single DataFrame
data = pd.concat([d1, d2, d3, d4], axis=1)
data.columns = ['RMSE', 'MAE', 'MAPE', 'SMAPE']

# Create the violin plot using seaborn
fig, axs = plt.subplots(1, 4, figsize=(8, 3.5))

sns.set_palette("flare")

# Create the violin plots and save them to the subplots
sns.violinplot(data=d1, ax=axs[0], color="burlywood")
axs[0].set_ylabel(data.columns[0])
axs[0].grid(True)
axs[0].set_axisbelow(True)
axs[0].yaxis.grid(color='gray', linestyle='dashed')

sns.violinplot(data=d2, ax=axs[1], color="sandybrown")
axs[1].set_ylabel(data.columns[1])
axs[1].grid(True)
axs[1].set_axisbelow(True)
axs[1].yaxis.grid(color='gray', linestyle='dashed')

sns.violinplot(data=d3, ax=axs[2], color="peru")
axs[2].set_ylabel(data.columns[2])
axs[2].grid(True)
axs[2].set_axisbelow(True)
axs[2].yaxis.grid(color='gray', linestyle='dashed')

sns.violinplot(data=d4, ax=axs[3], color="sienna")
axs[3].set_ylabel(data.columns[3])
axs[3].grid(True)
axs[3].set_axisbelow(True)
axs[3].yaxis.grid(color='gray', linestyle='dashed')

# Adjust the layout and save the figure
plt.tight_layout()
plt.savefig("violinplots.pdf", dpi=800, bbox_inches = 'tight')

"""# Benchmarking"""

# Load the dataset
horizon = 14
data_o = pd.read_csv("DADOS_HIDROLOGICOS_RES.csv", delimiter=';', decimal='.')
inflow_o = list(data_o['val_vazaoafluente'])
time = list(data_o['din_instante'])

x = inflow_o[(-365*3):-1]
f = pd.Series(x, index=pd.date_range(x[0], periods=len(x), freq="D"), name="DF")
filter = 'True'

if filter == 'True':
  x_resd_cf, x_trend_cf = sm.tsa.filters.cffilter(f, 2, 24, False)
  x_cf = list(x_trend_cf)
  x_resd_hp, x_trend_hp = sm.tsa.filters.hpfilter(f, 24)
  x_hp = list(x_trend_hp)
  x_trend_mtsl = MSTL(f, periods=(24, 24)).fit()
  x_mstl = list(x_trend_mtsl.trend)
  stl = STL(f, trend=23, seasonal=7)
  x_stl = list((stl.fit()).trend)

  # x = x_cf
  # x = x_hp
  x = x_stl
  # x = x_mstl

  time_data = pd.date_range(time[0], periods=len(x), freq='D')
  df = pd.DataFrame({'unique_id': np.array(['Airline1'] * len(x)),
      'ds': time_data, 'y': x, 'trend': np.arange(len(x)),})
  split_point = int(len(df) * 0.70)
  Y_train_df = df.iloc[:split_point].reset_index(drop=True)
  Y_test_df = df.iloc[split_point:].reset_index(drop=True)
  input_size = min(2 * horizon, len(Y_train_df))
else:
  time_data = pd.date_range(time[0], periods=len(x), freq='D')
  df = pd.DataFrame({'unique_id': np.array(['Airline1'] * len(x)),
      'ds': time_data, 'y': x, 'trend': np.arange(len(x)),})
  split_point = int(len(df) * 0.70)
  Y_train_df = df.iloc[:split_point].reset_index(drop=True)
  Y_test_df = df.iloc[split_point:].reset_index(drop=True)
  input_size = min(2 * horizon, len(Y_train_df))

import time
max_steps=100

start = time.time()

models = [TFT(input_size=horizon, h=horizon, max_steps=max_steps)]
nf = NeuralForecast(models=models, freq='d')
nf.fit(df=Y_train_df)
Y_hat_df = nf.predict().reset_index()

end = time.time()
time_s = end - start #save time

y_pred = Y_hat_df.TFT[:horizon]
y_true = Y_test_df.y[:horizon]

# RMSE & MAE & MAPE  & SMAPE
print(f'{(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(smape(y_true, y_pred)):.2E} & {time_s:.2f}')

import time
start = time.time()

models = [DilatedRNN (input_size=horizon, h=horizon, max_steps=max_steps)]
nf = NeuralForecast(models=models, freq='d')
nf.fit(df=Y_train_df)
Y_hat_df = nf.predict().reset_index()

end = time.time()
time_s = end - start #save time

y_pred = Y_hat_df.DilatedRNN [:horizon]
y_true = Y_test_df.y[:horizon]

# RMSE & MAE & MAPE  & SMAPE
print(f'{(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(smape(y_true, y_pred)):.2E} & {time_s:.2f}')

start = time.time()

models = [NHITS(input_size=horizon, h=horizon, max_steps=max_steps)]
nf = NeuralForecast(models=models, freq='d')
nf.fit(df=Y_train_df)
Y_hat_df = nf.predict().reset_index()

end = time.time()
time_s = end - start #save time

y_pred = Y_hat_df.NHITS[:horizon]
y_true = Y_test_df.y[:horizon]

# RMSE & MAE & MAPE  & SMAPE
print(f'{(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(smape(y_true, y_pred)):.2E} & {time_s:.2f}')

start = time.time()

models = [DeepNPTS(input_size=horizon, h=horizon, max_steps=max_steps)]
nf = NeuralForecast(models=models, freq='d')
nf.fit(df=Y_train_df)
Y_hat_df = nf.predict().reset_index()

end = time.time()
time_s = end - start #save time

y_pred = Y_hat_df.DeepNPTS[:horizon]
y_true = Y_test_df.y[:horizon]

# RMSE & MAE & MAPE  & SMAPE
print(f'{(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(smape(y_true, y_pred)):.2E} & {time_s:.2f}')

start = time.time()

models = [TCN(input_size=horizon, h=horizon, max_steps=max_steps)]
nf = NeuralForecast(models=models, freq='d')
nf.fit(df=Y_train_df)
Y_hat_df = nf.predict().reset_index()

end = time.time()
time_s = end - start #save time

y_pred = Y_hat_df.TCN[:horizon]
y_true = Y_test_df.y[:horizon]

# RMSE & MAE & MAPE  & SMAPE
print(f'{(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(smape(y_true, y_pred)):.2E} & {time_s:.2f}')

start = time.time()

models = [LSTM (input_size=horizon, h=horizon, max_steps=max_steps)]
nf = NeuralForecast(models=models, freq='d')
nf.fit(df=Y_train_df)
Y_hat_df = nf.predict().reset_index()

end = time.time()
time_s = end - start #save time

y_pred = Y_hat_df.LSTM[:horizon]
y_true = Y_test_df.y[:horizon]

# RMSE & MAE & MAPE  & SMAPE
print(f'{(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(smape(y_true, y_pred)):.2E} & {time_s:.2f}')

start = time.time()

models = [RNN (input_size=horizon, h=horizon, max_steps=max_steps)]
nf = NeuralForecast(models=models, freq='d')
nf.fit(df=Y_train_df)
Y_hat_df = nf.predict().reset_index()

end = time.time()
time_s = end - start #save time

y_pred = Y_hat_df.RNN[:horizon]
y_true = Y_test_df.y[:horizon]

# RMSE & MAE & MAPE  & SMAPE
print(f'{(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(smape(y_true, y_pred)):.2E} & {time_s:.2f}')

start = time.time()

models = [MLP (input_size=horizon, h=horizon, max_steps=max_steps)]
nf = NeuralForecast(models=models, freq='d')
nf.fit(df=Y_train_df)
Y_hat_df = nf.predict().reset_index()

end = time.time()
time_s = end - start #save time

y_pred = Y_hat_df.MLP[:horizon]
y_true = Y_test_df.y[:horizon]

# RMSE & MAE & MAPE  & SMAPE
print(f'{(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(smape(y_true, y_pred)):.2E} & {time_s:.2f}')

models = [NBEATS(input_size=horizon, h=horizon, max_steps=max_steps)]
nf = NeuralForecast(models=models, freq='d')
nf.fit(df=Y_train_df)
Y_hat_df = nf.predict().reset_index()

end = time.time()
time_s = end - start #save time

y_pred = Y_hat_df.NBEATS[:horizon]
y_true = Y_test_df.y[:horizon]

# RMSE & MAE & MAPE  & SMAPE
print(f'{(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(smape(y_true, y_pred)):.2E} & {time_s:.2f}')
