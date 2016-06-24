from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1/(y[ind]**2)
    return w

def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

dfxgb = joblib.load('pkl/dfxgb.pkl')
features = (2,4,5,7,9,11,12,13,14,15,17,28,29,30,31,32,33,34,35,36,37,38,39,40)
# Best
# 2,4,5,7,9,11,12,13,18,19,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
# (2,4,5,7,9,11,12,13,14,15,17,28,29,30,31,32,33,34,35,36,37,38,39,40)
# Not Good
# 2,4,5,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,28,29,30,31,32,33,34,35,36,37,38,39,40
X_train = dfxgb[dfxgb['Set']==1].iloc[:,features].values
X_test = dfxgb[dfxgb['Set']==2].iloc[:,features].values
y_train = dfxgb[dfxgb['Set']==1].iloc[:,6].values
y_test = dfxgb[dfxgb['Set']==2].iloc[:,6].values

rf = RandomForestRegressor()
rf.fit(X_train,y_train)
pred_y = rf.predict(X_test)

result = rmspe(pred_y,y_test)
print(result)

joblib.dump(pred_y,'pkl/pred_y.pkl')
joblib.dump(y_test,'pkl/y_test.pkl')