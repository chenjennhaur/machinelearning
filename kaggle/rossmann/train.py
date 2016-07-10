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
import random

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

def set_cv(data,best_features):
	data['CV'] = -1
	data.loc[(data['Date']>='01-Aug-2014')&(data['Date']<='17-Sep-2014'),'CV'] = 0
	data.loc[(data['Date']>='01-Aug-2013')&(data['Date']<='17-Sep-2013'),'CV'] = 1
	data.loc[(data['Date']>='01-Jun-2015')&(data['Date']<='17-Jul-2015'),'CV'] = 2
	X = data[data['Set']>0].loc[:,best_features].values
	y = data[data['Set']>0].iloc[:,6].values
	cv_set = data[data['Set']>0].iloc[:,41].values
	ps = PredefinedSplit(test_fold=cv_set)
	return (X,y,ps)

def parameter_tune(data,features,X,y,ps):
	param_grid = {
              "max_features": [5,20],
              "min_samples_split": [5,10],
              "min_samples_leaf": [5,10]
             }
			 
	model = RandomForestRegressor()
	rsearch = GridSearchCV(estimator=model,param_grid=param_grid,scoring=rmspe_scorer,cv=ps)
	rsearch.fit(X,y)
	
	# RandomizedSearchCV
	# param_grid = {
				  # "max_features": sp_randint(10, 20),
				  # "min_samples_split": sp_randint(1, 11),
				  # "min_samples_leaf": sp_randint(1, 11)
				 # }
	# model = RandomForestRegressor(n_jobs=-1)
	# rsearch = RandomizedSearchCV(estimator=model,param_distributions=param_grid,n_iter=30,scoring=rmspe_scorer,cv=3)
	# rsearch.fit(X,y)
	print(rsearch.best_score_)
	print(rsearch.best_estimator_)
	return rsearch.best_estimator_

def submission(model,X,y,X_test,labels):
	model.fit(X,y)
	pred_y = model.predict(X_test)
	submit_file = np.vstack((labels.T,pred_y.T))
	np.savetxt("submission.txt",submit_file.T,delimiter=",",fmt='%d',header='"Id","Sales"',comments='')

def select_features(model,data,ps):
	curr_features = []
	non_features = ('Customers','Date','Id','Set','Sales','CV')
	all_features = data.columns.tolist()
	min_score = 1000000
	random.shuffle(all_features)
	for f in all_features:
		# f = all_features.pop(1)
		if (f not in non_features) and (f not in curr_features) : 
			print("New Feature Added ",f)
			curr_features = list(set(curr_features + [f]))
			Xs = data[data['Set']>0].loc[:,curr_features].values
			ys = data[data['Set']>0].iloc[:,6].values
			score = -np.mean(cross_val_score(model,Xs,ys,scoring=rmspe_scorer,cv=ps))
			if (score < min_score): min_score = score
			if (score > min_score): curr_features.remove(f)
		print(min_score)
		print(curr_features)
	print("Minimum Score with features ",min_score)
	print("Best Features ",curr_features)
	return list(curr_features)

def onemodelperstore(data,best_features):
	cv = 2
	modelall = {}
	ypredict = []
	yactual = []
	# X_tr = data.loc[(data['CV']!=0,best_features]
	# y_tr = data.loc[(data['CV']!=0,best_features]
	for i in range(1,1115):
		#Training
		model = RandomForestRegressor()
		X_tr = data[(data['Set']>0)&(data['Store']==i)&(data['CV']!=cv)].loc[:,best_features].values
		y_tr = data[(data['Set']>0) &(data['Store']==i)&(data['CV']!=cv)].iloc[:,6].values
		model.fit(X_tr,y_tr)
		modelall[i] = model
		#Cross Validation
		X_cv = data.loc[(data['CV']==cv)&(data['Set']>0)&(data['Store']==i),best_features].values
		y_cv = data[(data['CV']==cv)&(data['Set']>0)&(data['Store']==i)].iloc[:,6].values
		y_pred = model.predict(X_cv)
		print("y_predict,y_actual")
		print(y_pred.shape," ",y_cv.shape)
		# print("y_predict",y_pred)
		# print("y_actual",y_cv)
		ypredict = np.hstack((ypredict,y_pred))
		yactual = np.hstack((yactual,y_cv)) 
		print("ypredict,yactual")
		print(ypredict.shape," ",yactual.shape)
		# print("y_predict_all",ypredict)
		# print("y_actual_all",yactual)
	
	print("Cross Validation", rmspe(yactual,ypredict))
	return modelall
	
	
	
rmspe_scorer = make_scorer(rmspe,greater_is_better=False)

dfxgb = joblib.load('pkl/dfxgb.pkl')
# pred_y = joblib.load('pkl/pred_y.pkl')

# best_features = (10,2,4,5,7,9,13,14,15,17,18,19,32,33,34,35,36,37,38,39,40)
best_features = ['Promo','CompetitionOpenSinceYear','CompetitionFlag','DateYear','DayOfWeek','stype_2','stype_3','CompetitionDistance','Store']
# Best
# (2,4,5,7,9,11,12,13,18,19,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40) - #0.133288201516
# (2,4,5,7,9,11,12,13,14,15,17,28,29,30,31,32,33,34,35,36,37,38,39,40) - #0.161455
# (10,2,5,7,9,13,14,15,17,18,19,32,33,34,35,36,37,38,39,40) - #0.15946
# (10,2,4,5,7,9,13,14,15,17,18,19,32,33,34,35,36,37,38,39,40) - # 0.156
# Not Good
# 2,4,5,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,28,29,30,31,32,33,34,35,36,37,38,39,40

X,y,ps = set_cv(dfxgb,best_features)
# rf = RandomForestRegressor(n_jobs=-1)
# best_features = select_features(rf,dfxgb,ps)	
clf = parameter_tune(dfxgb,best_features,X,y,ps)

# 1 model for each store
clf_store = onemodelperstore(dfxgb,best_features)

# Submission
# labels = dfxgb[dfxgb['Set']==0].iloc[:,3].values
# X_test = dfxgb[dfxgb['Set']==0].iloc[:,features].values
# submission(clf,X,y,X_test,labels)
	

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
