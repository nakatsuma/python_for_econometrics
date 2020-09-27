# -*- coding: utf-8 -*-
#%% NumPyの読み込み
import numpy as np
#   pandasの読み込み
import pandas as pd
#   statsmodelsの読み込み
import statsmodels.api as sm
#%% Rdatasetsからnswpsid1の読み込み
nsw = sm.datasets.get_rdataset('nsw74demo', 'DAAG')
nsw.data['age'] = (nsw.data['age'] - 18) / 10
nsw.data['age**2'] = nsw.data['age']**2
nsw.data['educ**2'] = nsw.data['educ']**2
nsw.data['re74'] = nsw.data['re74'] / 10000
nsw.data['re75'] = nsw.data['re75'] / 10000
nsw.data['re78'] = nsw.data['re78'] / 10000
nsw.data['re74**2'] = nsw.data['re74']**2
nsw.data['re75**2'] = nsw.data['re75']**2
n = nsw.data.shape[0]
re = nsw.data[['re78', 're75']].values.reshape((1, 2*n), order='F')
year78 = np.hstack((np.ones((1, n)), np.zeros((1, n))))
trained = np.tile(nsw.data['trt'].values, 2)
atet = year78 * trained
df = pd.DataFrame(data=np.vstack((re, trained, year78, atet)).T,
                  columns=['re', 'trained', 'year78', 'atet'])
#%% 処置群と対照群の標本平均の差
y_ate = nsw.data['re78']
X_ate = sm.add_constant(nsw.data['trt'])
results_ate = sm.OLS(y_ate, X_ate).fit(cov_type='HC1')
print(results_ate.summary())
#%% 処置群の職業訓練前後の標本平均の差
y_atet = df['re'].loc[df['trained'] == 1]
X_atet = sm.add_constant(df['year78'].loc[df['trained'] == 1])
results_atet = sm.OLS(y_atet, X_atet).fit(cov_type='HC1')
print(results_atet.summary())
#%% DID推定量
y_did = df['re']
X_did = sm.add_constant(df[['trained', 'year78', 'atet']])
results_did = sm.OLS(y_did, X_did).fit(cov_type='HC1')
print(results_did.summary())
#%% 傾向スコアをロジット・モデルで推定
varlist = ['educ', 'educ**2', 'age', 'age**2',
           'nodeg', 'black', 'hisp', 'marr',
           're74', 're74**2', 're75', 're75**2']
y_ps = nsw.data['trt']
X_ps = sm.add_constant(nsw.data[varlist])
results_ps = sm.Logit(y_ps, X_ps).fit()
print(results_ps.summary())
ps = results_ps.predict()
#%% IPW推定量
treated = nsw.data['trt']
weight = treated / ps + (1 - treated) / (1.0 - ps)
results_ipw = sm.WLS(y_ate, X_ate, weights=weight).fit(cov_type='HC1')
print(results_ipw.summary())
