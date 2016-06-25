import pandas as pd
import numpy as np
import time
import scipy.stats as spy
from datetime import datetime  
from sklearn.externals import joblib

### dtrain and dtest

dtrain = pd.read_csv('train.csv',
                     parse_dates=['Date'],
                     date_parser=(lambda dt:pd.to_datetime(dt,format='%Y-%m-%d')),
                     dtype={'Store':np.int16,'DayOfWeek':np.int8,'Sales':int,'Customers':np.int16,'Open':np.int8,'Promo':np.int8,'SchoolHoliday':np.int8})


dtest = pd.read_csv('test.csv',
                    dtype={'Id':int,'Store':np.int16,'DayOfWeek':np.int8,'Promo':np.int8,'SchoolHoliday':np.int8},
                    parse_dates=['Date'],
                    date_parser=(lambda dt:pd.to_datetime(dt,format='%Y-%m-%d')))


dtest['Open'].fillna(0,inplace=True)
dtest['Open'] = dtest['Open'].astype(int)

					
holiday_columns = ['0','a','b','c']
dtrain['StateHoliday'] = pd.Categorical(dtrain.StateHoliday, categories=holiday_columns)
dtest['StateHoliday'] = pd.Categorical(dtest.StateHoliday, categories=holiday_columns)

dtrain['Set'] = 1
dtest['Set'] = 0
### Combine train and test set
frames = [dtrain, dtest]
df = pd.concat(frames)

var_name = 'Date'
df[var_name + 'Day'] = pd.Index(df[var_name]).day.astype(np.int8)
df[var_name + 'Week'] = pd.Index(df[var_name]).week.astype(np.int8)
df[var_name + 'Month'] = pd.Index(df[var_name]).month.astype(np.int8)
df[var_name + 'Year'] = pd.Index(df[var_name]).year.astype(np.int16)
df[var_name + 'DayOfYear'] = pd.Index(df[var_name]).dayofyear.astype(np.int16)
df['DateInt'] = df['Date'].apply(lambda x: time.mktime(pd.to_datetime(x).timetuple())).astype(np.int32)
df['Set'] = df['Set'].astype(np.int8)
df['Open'] = df['Open'].astype(np.int8)
df['Promo'] = df['Promo'].astype(np.int8)
df['SchoolHoliday'] = df['SchoolHoliday'].astype(np.int8)

### dstore feature generation
dstore = pd.read_csv('store.csv',dtype={'Store':int})

for feature in dstore.columns:
    dstore[feature] = dstore[feature].fillna(0)

dstore['StoreType'] = dstore['StoreType'].astype('category').cat.codes
dstore['Assortment'] = dstore['Assortment'].astype('category').cat.codes
dstore['CompetitionDistance'] = dstore['CompetitionDistance'].astype(np.int32)
dstore['CompetitionOpenSinceYear'] = dstore['CompetitionOpenSinceYear'].astype(np.int16)
dstore['CompetitionOpenSinceMonth'] = dstore['CompetitionOpenSinceMonth'].astype(np.int8)
dstore['Promo2'] = dstore['Promo2'].astype(np.int8)
dstore['Promo2SinceWeek'] = dstore['Promo2SinceWeek'].astype(np.int8)
dstore['Promo2SinceYear'] = dstore['Promo2SinceYear'].astype(np.int16)

def convertCompetitionOpen(dsf):
    try:
        date = '{}-{}'.format(int(dsf['CompetitionOpenSinceYear']), int(dsf['CompetitionOpenSinceMonth']))
        return time.mktime(pd.to_datetime(date).timetuple())
    except:
        #return datetime.date(1970,1,1)
        return 0 

dstore['CompetitionOpenInt'] = dstore.apply(lambda dsf: convertCompetitionOpen(dsf), axis=1).astype(np.int32)

def convertPromo2(dsf):
    try:
        # %w = 1 refers to Monday
        date = '{}{}1'.format(int(dsf['Promo2SinceYear']), int(dsf['Promo2SinceWeek']))
#         return pd.to_datetime(date, format='%Y%W%w')
        return time.mktime(pd.to_datetime(date, format='%Y%W%w').timetuple())
    except:
#         return datetime.datetime(1970,1,1)
        return 0 

dstore['Promo2SinceFloat'] = dstore.apply(lambda dsf: convertPromo2(dsf), axis=1).astype(np.int32)

s = dstore['PromoInterval'].str.split(',').apply(pd.Series)
s.columns = ['PromoInterval0', 'PromoInterval1', 'PromoInterval2', 'PromoInterval3']
dstore = dstore.join(s)

monthToNum = {
            'Jan' : 1,
            'Feb' : 2,
            'Mar' : 3,
            'Apr' : 4,
            'May' : 5,
            'Jun' : 6,
            'Jul' : 7,
            'Aug' : 8,
            'Sept' : 9, 
            'Oct' : 10,
            'Nov' : 11,
            'Dec' : 12
} 

dstore['PromoInterval0'] = dstore['PromoInterval0'].apply(lambda x: monthToNum[x] if str(x) != 'nan' else 0).astype(np.int8)
dstore['PromoInterval1'] = dstore['PromoInterval1'].apply(lambda x: monthToNum[x] if str(x) != 'nan' else 0).astype(np.int8)
dstore['PromoInterval2'] = dstore['PromoInterval2'].apply(lambda x: monthToNum[x] if str(x) != 'nan' else 0).astype(np.int8)
dstore['PromoInterval3'] = dstore['PromoInterval3'].apply(lambda x: monthToNum[x] if str(x) != 'nan' else 0).astype(np.int8)

### Combine
dfm = pd.merge(df,dstore,how='left',on=['Store'])
for feature in dfm.columns:
    dfm[feature] = dfm[feature].fillna(0)

# Optimize size of dataframe
dfm.drop("Promo2",axis=1,inplace=True)
dfm.drop("PromoInterval",axis=1,inplace=True)
dfm['Sales'] = dfm['Sales'].astype(np.float32)
dfm = dfm.set_index(pd.DatetimeIndex(dfm['Date']))
dfm['Customers'] = dfm['Customers'].astype(np.int16)
dfm['Id'] = dfm['Id'].astype(np.uint16)

# New Features - Mean, Media, Harmonic Mean
std = dfm.groupby('Store').std()['Sales']
mean = dfm.groupby('Store').mean()['Sales']
median = dfm.groupby('Store').median()['Sales']
hmean = dfm[dfm['Sales']>0].groupby('Store').apply(lambda x: spy.hmean(x['Sales']))
pd_mean = pd.DataFrame(mean).reset_index()
pd_mean.columns = ['Store','Sales_Mean']
pd_median = pd.DataFrame(median).reset_index()
pd_median.columns = ['Store','Sales_Median']
pd_hmean = pd.DataFrame(hmean).reset_index()
pd_hmean.columns = ['Store','Sales_HMean']
pd_std = pd.DataFrame(std).reset_index()
pd_std.columns = ['Store','Sales_Std']
dfm = pd.merge(dfm,pd_mean,how='left',on=['Store'])
dfm = pd.merge(dfm,pd_median,how='left',on=['Store'])
dfm = pd.merge(dfm,pd_hmean,how='left',on=['Store'])
dfm = pd.merge(dfm,pd_std,how='left',on=['Store'])

def promo2Flag(df):
    if (df['Promo2SinceYear']==0):return 0
    date = '{}{}1'.format(int(df['Promo2SinceYear']), int(df['Promo2SinceWeek']))
    dt = pd.to_datetime(date, format='%Y%W%w')
    if (df['Date'] < dt) : return 0
    elif (df['Date'].month in list(df[['PromoInterval0','PromoInterval1','PromoInterval2','PromoInterval3']].values)): return 1
    else: return 0
dfm['promo2Flag'] = dfm.apply(lambda df: promo2Flag(df), axis=1).astype(np.int8)

def stateholidaytransform(df):
    if (df['StateHoliday']=='a'):return 1
    if (df['StateHoliday']=='b'):return 2
    if (df['StateHoliday']=='c'):return 3
    return 0  
dfm['StateHoliday'] = dfm.apply(lambda df: stateholidaytransform(df), axis=1).astype(np.int8)

def competitionFlag(df):
    if (pd.to_datetime(df['CompetitionOpenInt'],unit='s') < df['Date']): return 1
    else: return 0
dfm['CompetitionFlag'] = dfm.apply(lambda df: competitionFlag(df), axis=1).astype(np.int8)

# One Hot Vectors
sd_type = pd.get_dummies(dfm['StoreType'],prefix="stype")
sd_assort = pd.get_dummies(dfm['Assortment'],prefix='assort')
dfm = pd.concat([dfm,sd_type,sd_assort],axis=1)
dfm = dfm.drop(['Assortment'],axis=1).drop(['StoreType'],axis=1)
dfm['stype_0'] = dfm['stype_0'].astype(np.int8)
dfm['stype_1'] = dfm['stype_1'].astype(np.int8)
dfm['stype_2'] = dfm['stype_2'].astype(np.int8)
dfm['stype_3'] = dfm['stype_3'].astype(np.int8)
dfm['assort_0'] = dfm['assort_0'].astype(np.int8)
dfm['assort_1'] = dfm['assort_1'].astype(np.int8)
dfm['assort_2'] = dfm['assort_2'].astype(np.int8)

i = 0
for col in (dfm.columns):
    print(i,col)
    i += 1
	
joblib.dump(dfm,'pkl/dataset.pkl')
