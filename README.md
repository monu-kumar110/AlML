# evaluation metrics
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
 
# Actual values prediction
y_true = np.array([50,60,75,90,100])
 
# predicted values
y_pred = np.array([52,58,78,88,105])
 
# mae (mean absolute error)
mae = mean_absolute_error(y_true,y_pred)
print(mae)
 
# mse (mean squared error)
mse = mean_squared_error(y_true,y_pred)
print(mse)
 
# rmse (square root of mse)
rmse = np.sqrt(mse)
print(rmse)
 
# r2 score
r2 = r2_score(y_true,y_pred)
print(r2)