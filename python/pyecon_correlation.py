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
#%% 確率変数の生成
np.random.seed(99)
X1 = st.multivariate_normal.rvs(mean=np.zeros(2),
                                cov=np.array([[1.0, 0.9], [0.9, 1.0]]),
                                size=250)
X2 = st.multivariate_normal.rvs(mean=np.zeros(2),
                                cov=np.array([[1.0, -0.9], [-0.9, 1.0]]),
                                size=250)
X3 = st.multivariate_normal.rvs(mean=np.zeros(2),
                                cov=np.array([[1.0, 0.0], [0.0, 1.0]]),
                                size=250)
U = st.uniform.rvs(loc=0.0, scale=2.0*np.pi, size=250)
X = 2.0 * np.cos(U)
Y = 2.0 * np.sin(U)
#%% 確率変数間の相関の図示
fig, ax = plt.subplots(2, 2, sharex='all', sharey='all',
                       num=1, facecolor='w')
ax[0, 0].plot(X1[:,0], X1[:,1], 'k+')
ax[0, 0].axhline(color='k', linewidth=0.5)
ax[0, 0].axvline(color='k', linewidth=0.5)
ax[0, 0].set_xlim((-4.0, 4.0))
ax[0, 0].set_ylim((-4.0, 4.0))
ax[0, 0].set_ylabel('Y', fontproperties=jpfont)
ax[0, 0].set_title('正の相関 ($\\rho_{XY}$ = 0.9)', fontproperties=jpfont)
ax[0, 1].plot(X2[:,0], X2[:,1], 'k+')
ax[0, 1].axhline(color='k', linewidth=0.5)
ax[0, 1].axvline(color='k', linewidth=0.5)
ax[0, 1].set_title('負の相関 ($\\rho_{XY}$ = -0.9)', fontproperties=jpfont)
ax[1, 0].plot(X3[:,0], X3[:,1], 'k+')
ax[1, 0].axhline(color='k', linewidth=0.5)
ax[1, 0].axvline(color='k', linewidth=0.5)
ax[1, 0].set_xlabel('X', fontproperties=jpfont)
ax[1, 0].set_ylabel('Y', fontproperties=jpfont)
ax[1, 0].set_title('無相関 ($\\rho_{XY}$ = 0.0)', fontproperties=jpfont)
ax[1, 1].plot(X, Y, 'k+')
ax[1, 1].axhline(color='k', linewidth=0.5)
ax[1, 1].axvline(color='k', linewidth=0.5)
ax[1, 1].set_xlabel('X', fontproperties=jpfont)
ax[1, 1].set_title('非線形の関係 ($\\rho_{XY}$ = 0.0)', fontproperties=jpfont)
plt.savefig('pyecon_fig_correlation.png', dpi=300)
plt.show()
