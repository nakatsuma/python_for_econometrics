# -*- coding: utf-8 -*-
#%% NumPyの読み込み
import numpy as np
#   SciPyのstatsモジュールの読み込み
import scipy.stats as st
#   statsmodelsの読み込み
import statsmodels.api as sm
#   MatplotlibのPyplotモジュールの読み込み
import matplotlib.pyplot as plt
#   日本語フォントの設定
from matplotlib.font_manager import FontProperties
import sys
if sys.platform.startswith('win'):
    FontPath = 'C:\\Windows\\Fonts\\meiryo.ttc'
elif sys.platform.startswith('darwin'):
    FontPath = '/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc'
elif sys.platform.startswith('linux'):
    FontPath = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
else:
    print('このPythonコードが対応していないOSを使用しています．')
    sys.exit()
jpfont = FontProperties(fname=FontPath)
#%% Rdatasetsからpsid1の読み込み
psid1 = sm.datasets.get_rdataset('psid1', 'DAAG')
print(psid1.data)
#%% 単回帰モデルの推定
select_re75 = (psid1.data['re75'] > 0)
select_educ = (psid1.data['educ'] > 0) & (psid1.data['educ'] <= 16)
psid1.data = psid1.data[select_re75 & select_educ]
y = psid1.data['re75'] / 10000
x = psid1.data['educ']
X = sm.add_constant(x)
results = sm.OLS(y, X).fit(use_t=False)
print(results.summary())
#%% 散布図と回帰直線の描画
fig = plt.figure(num=1, facecolor='w')
a_hat, b_hat = results.params
x0 = np.linspace(0, 17, 18)
y_hat = results.predict(exog=sm.add_constant(x0))
plt.scatter(x, y, color='0.75', marker='+', label='データ')
plt.plot(x0, y_hat, 'k-',
         label='回帰直線 {0:<6.4f}+{1:<6.4f}$\\times$教育年数'.format(a_hat, b_hat))
plt.xlim((0, 17))
plt.ylim((0.0, 11.0))
plt.xlabel('教育年数', fontproperties=jpfont)
plt.ylabel('1975年の収入（万ドル）', fontproperties=jpfont)
plt.legend(loc='upper left', frameon=False, prop=jpfont)
plt.savefig('pyecon_fig_simple_regression.png', dpi=300)
plt.show()
