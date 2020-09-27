# -*- coding: utf-8 -*-
#%% pandasの読み込み
import pandas as pd
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
#%% 賃金曲線の描画
data = pd.read_csv('wage_curve.csv', index_col=0)
college = data['大学・大学院卒']
highschool = data['高校卒']
fig = plt.figure(num=1, facecolor='w')
plt.plot(college, 'k-', label='大学・大学院卒')
plt.plot(highschool, 'k--', label='高校卒')
plt.xlabel('年齢階級',  fontproperties=jpfont)
plt.ylabel('平均賃金（千円）', fontproperties=jpfont)
plt.legend(loc='upper left', frameon=False, prop=jpfont)
plt.savefig('pyecon_fig_wage_curve.png', dpi=300)
plt.show()
