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
#%% 二項分布
fig1 = plt.figure(num=1, facecolor='w')
bin_width = 0.25
grayscale = ['0.5', '0.75', '1.0']
n = 10
x = np.linspace(0, n, n + 1)
for idx, p in enumerate([0.2, 0.5, 0.8]):
    plt.bar(x + bin_width * (idx - 1.0), st.binom.pmf(x, n, p),
            width=bin_width, color=grayscale[idx], edgecolor='k',
            label='p = {0:<3.1f}'.format(p))
plt.xlim((-0.5, n + 0.5))
plt.ylim((0.0, 0.34))
plt.xlabel('確率変数', fontproperties=jpfont)
plt.ylabel('確率', fontproperties=jpfont)
plt.legend(loc='upper center', frameon=False, prop=jpfont)
plt.savefig('pyecon_fig_binomial.png', dpi=300)
plt.show()
#%% ポアソン分布
fig2 = plt.figure(num=2, facecolor='w')
bin_width = 0.25
grayscale = ['0.5', '0.75', '1.0']
x = np.linspace(0, 12, 13)
for idx, lam in enumerate([1.0, 3.0, 5.0]):
    plt.bar(x + bin_width * (idx - 1.0), st.poisson.pmf(x, lam),
            width=bin_width, color=grayscale[idx], edgecolor='k',
            label='$\\lambda$ = {0:<3.1f}'.format(lam))
plt.xlim((-0.5, 12.5))
plt.xlabel('確率変数', fontproperties=jpfont)
plt.ylabel('確率', fontproperties=jpfont)
plt.legend(loc='best', frameon=False, prop=jpfont)
plt.savefig('pyecon_fig_poisson.png', dpi=300)
plt.show()
#%% 一様分布
fig3 = plt.figure(num=3, facecolor='w')
grayscale = ['0.0', '0.35', '0.7']
for idx, par in enumerate([[0.0, 1.0], [-1.5, -0.5], [-2.0, 2.0]]):
    a_i = par[0]
    b_i = par[1]
    x_i = np.linspace(a_i, b_i, 11)
    plt.plot(x_i, st.uniform.pdf(x_i, loc=a_i, scale=b_i-a_i),
             color=grayscale[idx],
             label='[a,b] = [{0:<3.1f},{1:<3.1f}]'.format(a_i,b_i))
    plt.plot([a_i, a_i], [0.0, 1.0/(b_i-a_i)], 'k:', linewidth=0.5)
    plt.plot([b_i, b_i], [0.0, 1.0/(b_i-a_i)], 'k:', linewidth=0.5)
plt.xlim((-2.5, 2.5))
plt.ylim((0.0, 1.5))
plt.xlabel('確率変数', fontproperties=jpfont)
plt.ylabel('確率密度', fontproperties=jpfont)
plt.legend(loc='best', frameon=False, prop=jpfont)
plt.savefig('pyecon_fig_uniform.png', dpi=300)
plt.show()
#%% 指数分布
fig4 = plt.figure(num=4, facecolor='w')
grayscale = ['0.0', '0.35', '0.7']
x = np.linspace(0, 3, 101)
for idx, the in enumerate([1.0, 0.5, 2.0]):
    plt.plot(x, st.expon.pdf(x, scale=the),
             color=grayscale[idx],
             label='$\\theta$ = {0:<3.1f}'.format(the))
plt.xlim((0.0, 3.0))
plt.xlabel('確率変数', fontproperties=jpfont)
plt.ylabel('確率密度', fontproperties=jpfont)
plt.legend(loc='best', frameon=False, prop=jpfont)
plt.savefig('pyecon_fig_exponential.png', dpi=300)
plt.show()
#%% 正規分布
fig5, ax5 = plt.subplots(2, 1, sharex='col', num=5, facecolor='w')
grayscale = ['0.0', '0.35', '0.7']
x = np.linspace(-6.2, 6.2, 201)
ax5[0].set_xlim((-6.2, 6.2))
for idx, mu in enumerate([0.0, -2.0, 2.0]):
    ax5[0].plot(x, st.norm.pdf(x, loc=mu),
                color=grayscale[idx],
                label='$\\mu$ = {0:<3.1f}'.format(mu))
ax5[0].set_ylim((0.0, 0.45))
ax5[0].set_ylabel('確率密度', fontproperties=jpfont)
ax5[0].legend(loc='upper left', frameon=False)
for idx, sigma in enumerate([1.0, 0.5, 2.0]):
    ax5[1].plot(x, st.norm.pdf(x, scale=sigma),
                color=grayscale[idx],
                label='$\\sigma$ = {0:<3.1f}'.format(sigma))
ax5[1].set_ylim((0.0, 0.85))
ax5[1].set_xlabel('確率変数', fontproperties=jpfont)
ax5[1].set_ylabel('確率密度', fontproperties=jpfont)
ax5[1].legend(loc='upper left', frameon=False, prop=jpfont)
plt.savefig('pyecon_fig_gaussian.png', dpi=300)
plt.show()
#%% 二項分布の正規分布への収束
fig6, ax6 = plt.subplots(4, 1, sharex='col', num=6, facecolor='w')
ax6[0].set_xlim((-0.5, 20.5))
ax6[0].set_xticks(np.linspace(0, 20, 11))
for idx, n in enumerate([5, 20, 50, 100]):
    x = np.linspace(0, n, n + 1)
    ax6[idx].bar(x, st.binom.pmf(x, n, 0.1),
                 color='0.5', edgecolor='k',
                 label='n = {0:< 3.0f}'.format(n))
    ax6[idx].set_ylabel('確率密度', fontproperties=jpfont)
    ax6[idx].legend(loc='upper right', frameon=False, prop=jpfont)
ax6[-1].set_xlabel('確率変数', fontproperties=jpfont)
plt.savefig('pyecon_fig_binomial2gaussian.png', dpi=300)
plt.show()
