# AI_Project_3
AI_Project_3

__Purpose__: the purpose of this project is to nalyze historical unit sales and identify modeling technique that optimizes forecast

&nbsp; &nbsp; __Step 1__: Preprocess the data  
&nbsp; &nbsp; __Step 2__: CCreate, Compile, and Train the Model  
&nbsp; &nbsp;__Step 3__: Provide a summary and answer questions regarding the model and selections

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
