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
#%% 大数の法則
np.random.seed(99)
fig1 = plt.figure(num=1, facecolor='w')
grayscale1 = ['0.0', '0.3', '0.5', '0.7']
a = -np.sqrt(3.0)
b_a = 2.0 * np.sqrt(3.0)
m = 100000
for idx, n in enumerate([20, 80, 320, 1280]):
    sample = st.uniform.rvs(loc=a, scale=b_a, size=(m, n))
    sample_mean = sample.mean(axis=1)
    plt.hist(sample_mean, density=True, histtype='step', bins=41,
             color=grayscale1[idx], label='n = {0:<4.0f}'.format(n))
plt.xlim((-0.7, 0.7))
plt.xlabel('$\\bar{X}_n - \\mu$')
plt.legend(loc='best', frameon=False, prop=jpfont)
plt.savefig('pyecon_fig_lln.png', dpi=300)
plt.show()
#%% 中心極限定理
np.random.seed(99)
fig2 = plt.figure(num=2, facecolor='w')
grayscale2 = ['0.3', '0.5', '0.7']
x = np.linspace(-6.0, 6.0, 201)
plt.plot(x, st.norm.pdf(x), 'k-', label='標準正規分布')
for idx, n in enumerate([2, 4, 50]):
    sample = st.uniform.rvs(loc=a, scale=b_a, size=(m, n))
    sample_mean = sample.mean(axis=1)
    se = sample.std(axis=1, ddof=1) / np.sqrt(n)
    t_value = sample_mean / se
    plt.hist(t_value, density=True, histtype='step', bins=81,
             range=(-6.0, 6.0), color=grayscale2[idx],
             label='n = {0:<2.0f}'.format(n))
plt.xlim((-6.0, 6.0))
plt.xlabel('$(\\bar{X}_n - \\mu)/(S_n/\\sqrt{n})$')
plt.legend(loc='best', frameon=False, prop=jpfont)
plt.savefig('pyecon_fig_clt.png', dpi=300)
plt.show()
