# -*- coding: utf-8 -*-
#%% NumPyの読み込み
import numpy as np
#   SciPyのstatsモジュールの読み込み
import scipy.stats as st
#   statsmodelsの読み込み
import statsmodels.api as sm
#   2段階最小2乗法を実行するIV2SLSとGMMを実行するIVGMMの読み込み
from statsmodels.sandbox.regression.gmm import IV2SLS, IVGMM
#%% RdatasetsからMrozの読み込み
mroz = sm.datasets.get_rdataset('Mroz', 'Ecdat')
mroz.data = mroz.data[mroz.data['hearnw'] > 0]
print(st.pearsonr(mroz.data['educw'], mroz.data['educwf']))
print(st.pearsonr(mroz.data['educw'], mroz.data['educwm']))
#%% 収入を教育年数で説明する単回帰モデル
y = np.log(mroz.data['hearnw'])
x = mroz.data[['educw']]
X = sm.add_constant(x)
results_ols = sm.OLS(y, X).fit(use_t=False)
print(results_ols.summary())
#%% 父母の教育年数を操作変数として使う2SLS
z = mroz.data[['educwf', 'educwm']]
Z = sm.add_constant(z)
results_iv = IV2SLS(y, X, instrument=Z).fit()
print(results_iv.summary())
#%% 2SLSの代わりにGMMでIV推定量を求める
results_gmm = IVGMM(y, X, instrument=Z).fit()
print(results_gmm.summary())
