# -*- coding: utf-8 -*-
#%% NumPyの読み込み
import numpy as np
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
#%% マルコフ連鎖における確率ベクトルの計算
transition_matrix = np.array([[0.9, 0.25], [0.1, 0.75]])
initial_probability = np.array([0.5, 0.5])
time_horizon = 11
markov_chain = np.empty((time_horizon, 2))
markov_chain[0, :] = initial_probability
state_probability = initial_probability
for t in range(1, time_horizon):
    state_probability = transition_matrix @ state_probability
    markov_chain[t, :] = state_probability
#%% マルコフ連鎖の収束のグラフの作成
fig = plt.figure(num=1, facecolor='w')
periods = np.linspace(0, time_horizon - 1, time_horizon)
plt.plot(periods, markov_chain[:, 0], 'k-x', label='拡張')
plt.plot(periods, markov_chain[:, 1], 'k:.', label='後退')
plt.xlim((periods.min(), periods.max()))
plt.ylim((0.0, 1.0))
plt.xlabel('時点', fontproperties=jpfont)
plt.ylabel('確率', fontproperties=jpfont)
plt.legend(loc='best', frameon=False, prop=jpfont)
plt.savefig('pyecon_fig_markov_chain.png', dpi=300)
plt.show()
