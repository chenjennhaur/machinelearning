# 0 Customers
# 1 Date
# 2 DayOfWeek
# 3 Id
# 4 Open
# 5 Promo
# 6 Sales
# 7 SchoolHoliday
# 8 Set
# 9 StateHoliday
# 10 Store
# 11 DateDay
# 12 DateWeek
# 13 DateMonth
# 14 DateYear
# 15 DateDayOfYear
# 16 DateInt
# 17 CompetitionDistance
# 18 CompetitionOpenSinceMonth
# 19 CompetitionOpenSinceYear
# 20 Promo2SinceWeek
# 21 Promo2SinceYear
# 22 CompetitionOpenInt
# 23 Promo2SinceFloat
# 24 PromoInterval0
# 25 PromoInterval1
# 26 PromoInterval2
# 27 PromoInterval3
# 28 Sales_Mean (By Store)
# 29 Sales_Median (By Store)
# 30 Sales_HMean (By Store)
# 31 Sales_Std (By Store)
# 32 stype_0
# 33 stype_1
# 34 stype_2
# 35 stype_3
# 36 assort_0
# 37 assort_1
# 38 assort_2
# 39 promo2Flag
# 40 CompetitionFlag

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import make_scorer
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint
import numpy as np

def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1/(y[ind]**2)
    return w

def rmspe(ground_truth,prediction):
    #w = ToWeight(y)
	w = np.zeros(ground_truth.shape, dtype=float)
	ind = ground_truth != 0
	w[ind] = 1/(ground_truth[ind]**2)
	return np.sqrt(np.mean( w * (ground_truth-prediction)**2 ))

rmspe_scorer = make_scorer(rmspe,greater_is_better=False)
	
dfxgb = joblib.load('pkl/dfxgb.pkl')
# pred_y = joblib.load('pkl/pred_y.pkl')
features = (10,2,4,5,7,9,13,14,15,17,18,19,32,33,34,35,36,37,38,39,40)
# Best
# (2,4,5,7,9,11,12,13,18,19,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40) - #0.133288201516
# (2,4,5,7,9,11,12,13,14,15,17,28,29,30,31,32,33,34,35,36,37,38,39,40) - #0.161455
# (10,2,5,7,9,13,14,15,17,18,19,32,33,34,35,36,37,38,39,40) - #0.15946
# (10,2,4,5,7,9,13,14,15,17,18,19,32,33,34,35,36,37,38,39,40) - # 0.156
# Not Good
# 2,4,5,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,28,29,30,31,32,33,34,35,36,37,38,39,40

### Cross Validation code
X_train = dfxgb[dfxgb['Set']==1].iloc[:,features].values
# X_test = dfxgb[dfxgb['Set']==2].iloc[:,features].values
y_train = dfxgb[dfxgb['Set']==1].iloc[:,6].values
# y_test = dfxgb[dfxgb['Set']==2].iloc[:,6].values

param_grid = {
              "max_features": sp_randint(10, 20),
              "min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11)
             }
model = RandomForestRegressor(n_jobs=4)
rsearch = RandomizedSearchCV(estimator=model,param_distributions=param_grid,n_iter=30,scoring=rmspe_scorer,cv=5)
rsearch.fit(X_train,y_train)
print(rsearch.best_score_)
print(rsearch.best_estimator_)

# rf = RandomForestRegressor(bootstrap=True,criterion='mse',max_depth=None,max_features=10,max_leaf_nodes=None,min_samples_leaf=4,min_samples_split=8,min_weight_fraction_leaf=0,n_estimators=8,n_jobs=4,oob_score=False,random_state=None,verbose=0,warm_start=False)
# rf.fit(X_train,y_train)
# pred_y = rf.predict(X_test)
# result = rmspe(y_test,pred_y)

# print(rmspe(y_train,rf.predict(X_train)))
# print(rmspe_scorer(rf,X_train,y_train))

# np.savetxt("X_train.csv",X_train,delimiter=",")
# np.savetxt("X_test.csv",X_test,delimiter=",")
# np.savetxt("y_train.csv",y_train,delimiter=",")
# np.savetxt("y_test.csv",y_test,delimiter=",")

# joblib.dump(pred_y,'pkl/pred_y.pkl')
# joblib.dump(y_test,'pkl/y_test.pkl')

### Submission code (Ctrl-K,Q)

# X_train = joblib.load('pkl/X_train.pkl')
# y_train = joblib.load('pkl/y_train.pkl')
# X_test = joblib.load('pkl/X_test.pkl')
# labels = joblib.load('pkl/labels.pkl')

# X_train = dfxgb[dfxgb['Set']>0].iloc[:,features].values
# y_train = dfxgb[dfxgb['Set']>0].iloc[:,6].values
# X_test = dfxgb[dfxgb['Set']==0].iloc[:,features].values
# labels = dfxgb[dfxgb['Set']==0].iloc[:,3].values

# rf = RandomForestRegressor(n_estimators=8)
# rf.fit(X_train,y_train)

# pred_y = rf.predict(X_test)
# submission = np.vstack((labels.T,pred_y.T))
# np.savetxt("submission.txt",submission.T,delimiter=",",fmt='%d',header='"Id","Sales"',comments='')

# For import to R
# np.savetxt("X_train.csv",X_train,delimiter=",",fmt='%d')
# np.savetxt("y_train.csv",y_train,delimiter=",",fmt='%d')
# np.savetxt("X_test.csv",X_test,delimiter=",",fmt='%d')
# np.savetxt("labels.csv",labels,delimiter=",",fmt='%d')


# joblib.dump(X_train,'pkl/X_train.pkl')
# joblib.dump(y_train,'pkl/y_train.pkl')
# joblib.dump(X_test,'pkl/X_test.pkl')
# joblib.dump(labels,'pkl/labels.pkl')
# joblib.dump(pred_y,'pkl/pred_y.pkl')
# joblib.dump(rf,'pkl/rf.pkl')

