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
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import PredefinedSplit
from sklearn.cross_validation import cross_val_score
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

### Parameter Tuning

def set_cv(data):
	data['CV'] = -1
	data.loc[(data['Date']>='01-Aug-2014')&(data['Date']<='17-Sep-2014'),'CV'] = 0
	data.loc[(data['Date']>='01-Aug-2013')&(data['Date']<='17-Sep-2013'),'CV'] = 1
	data.loc[(data['Date']>='01-Jun-2015')&(data['Date']<='17-Jul-2015'),'CV'] = 2
	X = data[data['Set']>0].iloc[:,features].values
	y = data[data['Set']>0].iloc[:,6].values
	cv_set = data[data['Set']>0].iloc[:,41].values
	ps = PredefinedSplit(test_fold=cv_set)
	return (X,y,ps)

def parameter_tune(data,features,X,y,ps):
	param_grid = {
              "max_features": [5,10],
              "min_samples_split": [10],
              "min_samples_leaf": [10]
             }
			 
	model = RandomForestRegressor(n_jobs=-1)
	rsearch = GridSearchCV(estimator=model,param_grid=param_grid,scoring=rmspe_scorer,cv=ps)
	rsearch.fit(X,y)
	
	# RandomizedSearchCV
	# param_grid = {
				  # "max_features": sp_randint(10, 20),
				  # "min_samples_split": sp_randint(1, 11),
				  # "min_samples_leaf": sp_randint(1, 11)
				 # }
	# model = RandomForestRegressor(n_jobs=4)
	# rsearch = RandomizedSearchCV(estimator=model,param_distributions=param_grid,n_iter=30,scoring=rmspe_scorer,cv=5)
	# rsearch.fit(X_train,y_train)
	print(rsearch.best_score_)
	print(rsearch.best_estimator_)
	return rsearch.best_estimator_

def submission(model,X,y,X_test,labels):
	model.fit(X,y)
	pred_y = model.predict(X_test)
	submit_file = np.vstack((labels.T,pred_y.T))
	np.savetxt("submission.txt",submit_file.T,delimiter=",",fmt='%d',header='"Id","Sales"',comments='')

def select_features(model,data,feature_list,X,y,ps):
	score = np.mean(cross_val_score(model,X,y,scoring=rmspe_scorer,cv=ps))
	return score
	
	
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

X,y,ps = set_cv(dfxgb)
rf = RandomForestRegressor()
# select_features(rf,dfxgb,(2),X,y,ps)	
clf = parameter_tune(rf,dfxgb,features,X,y,ps)

# Submission
labels = dfxgb[dfxgb['Set']==0].iloc[:,3].values
X_test = dfxgb[dfxgb['Set']==0].iloc[:,features].values
submission(clf,X,y,X_test,labels)
	

### Comment (Ctrl-K,Q)

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

