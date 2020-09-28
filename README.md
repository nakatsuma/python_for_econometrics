# 中妻照雄「Pythonによる計量経済学入門」

[朝倉書店ウェブサイト](http://www.asakura.co.jp/books/isbn/978-4-254-12899-4/ "朝倉書店ウェブサイト")

---

+ [Pythonのインストール手順](#pythonのインストール手順)
+ [Pythonコード](#pythonコード)
  + [第2章](#第2章)
  + [第3章](#第3章)
  + [第4章](#第4章)
  + [第5章](#第5章)

---

## Pythonのインストール手順

1. 古いAnacondaがインストールされているときは、この[手順](https://docs.anaconda.com/anaconda/install/uninstall/)
でアンインストールしておく。

2. Anacondaのインストーラー (Windows, macOS or Linux) を[ここ](https://www.anaconda.com/products/individual)から入手する.

3. ダウンロードしたインストーラーをダブルクリックして Anacondaのインストールを行う。

## Pythonコード

### 第2章

+ コード2.1 マルコフ連鎖の収束: [pyecon\_markov\_chain.py](python/pyecon_markov_chain.py)

### 第3章

+ コード3.1 代表的な確率分布: [pyecon\_distribution.py](python/pyecon_distribution.py)
+ コード3.2 歪度と尖度: [pyecon\_skewness\_kurtosis.py](python/pyecon_skewness_kurtosis.py)
+ コード3.3 確率変数間の相関関係: [pyecon\_correlation.py](python/pyecon_correlation.py)
+ コード3.4 大数の法則と中心極限定理: [pyecon\_large\_sample.py](python/pyecon_large_sample.py)

### 第4章

+ コード4.1 単回帰モデルの最小2乗法による推定: [pyecon\_simple\_regression.py](python/pyecon_simple_regression.py)
+ コード4.2 回帰係数のOLS推定量とt統計量の分布: [pyecon\_ols\_simulation.py](python/pyecon_ols_simulation.py)
+ コード4.3 年齢階級別賃金曲線の作図: [pyecon\_wage\_curve.py](python/pyecon_wage_curve.py)
+ 年齢階級別賃金データ: [wage\_curge.csv](python/wage_curve.csv)

### 第5章

+ コード5.1 収入の重回帰モデルの推定: [pyecon\_multiple\_regression.py](python/pyecon_multiple_regression.py)
+ コード5.2 操作変数法の例: [pyecon\_iv.py](python/pyecon_iv.py)
+ コード5.3 職業訓練の処置効果の検証: [pyecon\_job\_training.py](python/pyecon_job_training.py)
