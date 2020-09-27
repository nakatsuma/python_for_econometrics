# -*- coding: utf-8 -*-
#%% NumPyの読み込み
import numpy as np
#   statsmodelsの読み込み
import statsmodels.api as sm
#%% Rdatasetsからpsid1の読み込み
psid1 = sm.datasets.get_rdataset('psid1', 'DAAG')
select_re75 =  (psid1.data['re75'] > 0)
select_educ =  (psid1.data['educ'] > 0) & (psid1.data['educ'] <= 16)
psid1.data = psid1.data[select_re75 & select_educ]
psid1.data['age'] = (psid1.data['age'] - 18) / 10
psid1.data['age**2'] = psid1.data['age']**2
y = psid1.data['re75'] / 10000
#%% 仮説検定用の制約付き回帰モデルの推定
results0 = sm.OLS(y, np.ones(len(y))).fit(use_t=False)
results = sm.OLS(y, sm.add_constant(psid1.data['educ'])).fit(use_t=False)
#%% 年齢の効果を含めた重回帰モデルの推定
varset1 = ['educ', 'age', 'age**2']
X1 = sm.add_constant(psid1.data[varset1])
results1 = sm.OLS(y, X1).fit(use_t=False)
print(results1.summary())
#%% ダミー変数を加えた重回帰モデルの推定
varset2 = ['educ', 'age', 'age**2', 'nodeg', 'black', 'hisp', 'marr']
X2 = sm.add_constant(psid1.data[varset2])
results2 = sm.OLS(y, X2).fit(use_t=False)
print(results2.summary())
