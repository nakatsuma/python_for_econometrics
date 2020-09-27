# -*- coding: utf-8 -*-
#%% NumPyの読み込み
import numpy as np
#   SciPyのstatsモジュールの読み込み
import scipy.stats as st
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
#%% OLS推定量などの計算を行う関数
def ols_simulation(coef, n, m, u):
    """
        入力
        coef:   パラメータの真の値
        n:      標本の大きさ
        m:      データの生成を繰り返す回数
        u:      誤差項のデータ
        出力
        OLS推定量，t統計量，決定係数を含む辞書
    """
    x = st.uniform.rvs(loc=-6.0, scale=12.0, size=(n, m))
    y = coef[0] + coef[1] * x + u
    y_bar = np.mean(y, axis=0)
    x_bar = np.mean(x, axis=0)
    d_y = y - np.tile(y_bar, (n, 1))
    d_x = x - np.tile(x_bar, (n, 1))
    v_x = np.sum(d_x**2, axis=0)
    beta_hat = np.sum(d_x * d_y, axis=0) / v_x
    alpha_hat = y_bar - beta_hat * x_bar
    e = y - np.tile(alpha_hat, (n, 1)) - np.tile(beta_hat, (n, 1)) * x
    rss = np.sum(e**2, axis=0)
    s_squared = rss / (n - 2)
    se_alpha = np.sqrt((s_squared / n) * (np.sum(x**2, axis=0) / v_x))
    se_beta = np.sqrt(s_squared / v_x)
    t_alpha = (alpha_hat - coef[0]) / se_alpha
    t_beta = (beta_hat - coef[1]) / se_beta
    r_squared = 1.0 - rss / np.sum(d_y**2, axis=0)
    sim_result = {'alpha_hat':alpha_hat, 'beta_hat':beta_hat,
                  't_alpha':t_alpha, 't_beta':t_beta,
                  's_squared':s_squared, 'r_squared':r_squared}
    return sim_result
#%% 単回帰モデルからの人工データの生成とOLS推定量などの計算
np.random.seed(99)
n = 50
m = 100000
distribution = ['一様分布', '指数分布', '正規分布']
error = [st.uniform.rvs(loc=-np.sqrt(3.0), scale=2.0*np.sqrt(3.0),
                        size=(n, m)),
         st.expon.rvs(loc=-1.0, size=(n, m)),
         st.norm.rvs(size=(n, m))]
h = len(error)
coef = np.array([0.0, 1.0])
results = dict()
for idx in range(h):
     results[distribution[idx]] = ols_simulation(coef, n, m, error[idx])
#%% OLS推定量のヒストグラムの作図
fig1, ax1 = plt.subplots(3, h, num=1, facecolor='w',
                         sharex='row', sharey='row')
ax1[0, 0].set_ylabel('$\\alpha$の推定量', fontproperties=jpfont)
ax1[1, 0].set_ylabel('$\\beta$の推定量', fontproperties=jpfont)
ax1[2, 0].set_ylabel('$\\sigma^2$の推定量', fontproperties=jpfont)
ax1[0, 0].set_xlim((-0.6, 0.6))
ax1[1, 0].set_xlim((0.85, 1.15))
ax1[2, 0].set_xlim((0.0, 2.6))
for idx in range(h):
    ax1[0, idx].set_title(distribution[idx], fontproperties=jpfont)
    ax1[0, idx].hist(results[distribution[idx]]['alpha_hat'],
                     density=True, histtype='step', bins=31, color='k')
    ax1[1, idx].hist(results[distribution[idx]]['beta_hat'],
                     density=True, histtype='step', bins=31, color='k')
    ax1[2, idx].hist(results[distribution[idx]]['s_squared'],
                     density=True, histtype='step', bins=31, color='k')
plt.tight_layout()
plt.savefig('pyecon_fig_ols_estimator.png', dpi=300)
plt.show()
#%% t統計量のヒストグラムの作図
fig2, ax2 = plt.subplots(3, h, num=2, facecolor='w',
                         sharex='row', sharey='row')
ax2[0, 0].set_ylabel('$\\alpha$のt統計量', fontproperties=jpfont)
ax2[1, 0].set_ylabel('$\\beta$のt統計量', fontproperties=jpfont)
ax2[2, 0].set_ylabel('決定係数$R^2$', fontproperties=jpfont)
ax2[0, 0].set_xlim((-4.5, 4.5))
ax2[1, 0].set_xlim((-4.5, 4.5))
ax2[2, 0].set_xlim((0.8, 1.0))
for idx in range(h):
    ax2[0, idx].set_title(distribution[idx], fontproperties=jpfont)
    ax2[0, idx].hist(results[distribution[idx]]['t_alpha'],
                     density=True, histtype='step', bins=31, color='k')
    ax2[1, idx].hist(results[distribution[idx]]['t_beta'],
                     density=True, histtype='step', bins=31, color='k')
    ax2[2, idx].hist(results[distribution[idx]]['r_squared'],
                     density=True, histtype='step', bins=31, color='k')
plt.tight_layout()
plt.savefig('pyecon_fig_t_statistic.png', dpi=300)
plt.show()
