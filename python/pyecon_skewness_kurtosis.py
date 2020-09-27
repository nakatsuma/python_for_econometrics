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
#%% 歪度と尖度
from scipy.special import gamma
fig, ax = plt.subplots(2, 1, num=1, facecolor='w')
grayscale = ['0.0', '0.35', '0.7']
x = np.linspace(-5.0, 7.0, 251)
label1 = ['左右対称な分布', '正に歪んだ分布', '負に歪んだ分布']
par1 = [[0.0, 0.0, 1.0], [8.0, -3.0, 2.0], [-8.0, 3.0, 2.0]]
for idx, par in enumerate(par1):
    ax[0].plot(x, st.skewnorm.pdf(x, par[0], loc=par[1], scale=par[2]),
                color=grayscale[idx], label=label1[idx])
ax[0].set_xlim((-5.0, 7.0))
ax[0].set_ylim((0.0, 0.42))
ax[0].set_ylabel('確率密度', fontproperties=jpfont)
ax[0].legend(loc='upper right', frameon=False, prop=jpfont)
label2 = ['正規分布', '裾の厚い分布', '裾の薄い分布']
par2 = [2.0, 1.0, 8.0]
for idx, par in enumerate(par2):
    inv_sd = np.sqrt(gamma(1.0 / par) / gamma(3.0 / par))
    ax[1].plot(x, st.gennorm.pdf(x, par, scale=inv_sd),
                color=grayscale[idx], label=label2[idx])
ax[1].set_xlim((-4.5, 4.5))
ax[1].set_ylim((0.0, 0.73))
ax[1].set_xlabel('確率変数', fontproperties=jpfont)
ax[1].set_ylabel('確率密度', fontproperties=jpfont)
ax[1].legend(loc='upper right', frameon=False, prop=jpfont)
plt.savefig('pyecon_fig_skewness_kurtosis.png', dpi=300)
plt.show()
