# AI_Project_3
AI_Project_3

__Purpose__: the purpose of this project is to nalyze historical unit sales and identify modeling technique that optimizes forecast

&nbsp; &nbsp; __Step 1__: Pull hisotrical item level unit sales data from exisiting company database 
&nbsp; &nbsp; __Step 2__: Preprocess the Data   
&nbsp; &nbsp; &nbsp;__*__: Change datatypes where necessary    
&nbsp; &nbsp; &nbsp;__*__: Remove 2017 data as it did not contain all metrics
&nbsp; &nbsp;__Step 3__: Graph the unit sales history for each brand to show the complexity of the data and show seasonality changes where relevant  
&nbsp; &nbsp;__Step 4__: Filter data to a single UPC  
&nbsp; &nbsp;__Step 5__: Calculate the .diff of unit sales to use for modeling, add to the dataframe  
&nbsp; &nbsp;__Step 6__: Set up, train, tune pytorch based LSTM model  
&nbsp; &nbsp; &nbsp;__*__: Used 90% of chosed UPC data for training since the full 6 years of history was available  
&nbsp; &nbsp; &nbsp;__*__: 1 layer nueral network with 2000 epochs produced the best results (tested with 1-6 layers, and from 500-3000 epochs)    
&nbsp; &nbsp; &nbsp;__*__: printed test results     
&nbsp; &nbsp; &nbsp;__*__: processed test results in excel and compared them to existing model performance  
&nbsp; &nbsp;__Step 7__: Set up, train, tune persistence Long Short-term model  
&nbsp; &nbsp; &nbsp;__*__: training set was based on all weeks up to the selected forecasted period of 16 weeks  
&nbsp; &nbsp; &nbsp;__*__: 4 layer nueral network with 3000 epochs produced the best results (tested with 1-6 layers, and from 500-3500 epochs)  
&nbsp; &nbsp; &nbsp;__*__: printed test results  
&nbsp; &nbsp; &nbsp;__*__: processed test results in excel and compared them to existing model performance  
&nbsp; &nbsp;__Step 8__: Set up, train, Linear Regression model  
&nbsp; &nbsp; &nbsp;__*__: Additional pre-processing data for Linear, Random Forest, and XG Boost models  
&nbsp; &nbsp; &nbsp;__*__: training set was based on all weeks up to the selected forecasted period of 16 weeks  
&nbsp; &nbsp; &nbsp;__*__: printed test results  
&nbsp; &nbsp; &nbsp;__*__: processed test results in excel and compared them to existing model performance  
&nbsp; &nbsp;__Step 9__: Set up, train, Random Forest model  
&nbsp; &nbsp; &nbsp;__*__: training set was based on all weeks up to the selected forecasted period of 16 weeks  
&nbsp; &nbsp; &nbsp;__*__: printed test results  
&nbsp; &nbsp; &nbsp;__*__: processed test results in excel and compared them to existing model performance  
&nbsp; &nbsp;__Step 10__: Set up, train, XG Boost model  
&nbsp; &nbsp; &nbsp;__*__: training set was based on all weeks up to the selected forecasted period of 16 weeks  
&nbsp; &nbsp; &nbsp;__*__: printed test results  
&nbsp; &nbsp; &nbsp;__*__: processed test results in excel and compared them to existing model performance

__Intalls__: A number of installations are needed (and provided) to run the code necessary to complete this project    
  
    
      
from datetime import datetime  
import os  

import pandas as pd  
from pandas import concat  

  
import numpy as np  
from numpy import array  
  
import sklearn as skl  
from sklearn.ensemble import RandomForestRegressor  
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import MinMaxScaler  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  
from sklearn.model_selection import train_test_split  

from xgboost import XGBRegressor  
  
import matplotlib as mpl  
import matplotlib.pyplot as plt  
from matplotlib import pyplot  
  
import tensorflow as tf  
from tensorflow.keras import layers, models, Model  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import LSTM, Dense  
from tensorflow import keras  
from tensorflow.keras.layers import LSTM  
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  
  
import seaborn as sns  
  
  import keras_tuner as kt  
from keras.models import Sequential  
from keras.layers import Dense  
  
  import torch  
  import torch.nn as nn  
  import torch.optim as optim  
  import torch.utils.data as data  
    
  from math import sqrt

[Python Documentation] (https://docs.python.org/)  
[Jupyter Notebook Documentation] (https://jupyter-notebook.readthedocs.io/)  
[sklearn Documentation]  (https://scikit-learn.org/0.21/documentation.html)  
[tensorflow Documentation]  (https://devdocs.io/tensorflow~2.4/)  
[Pytoch Documentation] (https://pytorch.org/docs/stable/index.html)  
[Matplot Documentation] (https://matplotlib.org/stable/users/index.html)  
[XG Boost Documentation] (https://xgboost.readthedocs.io/en/stable/)

__Resources__:  
[Xpert Learning Assistance]   
[README.md formatting] (https://medium.com/analytics-vidhya/writing-github-readme-e593f278a796)  
