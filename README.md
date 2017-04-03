## サマリ
- URL: https://www.kaggle.com/c/leaf-classification
- 画像と特徴量(192次元)から99クラスのラベルに葉っぱを分類する．

## Score
- 0.06235

## Datasets
- 192次元の特徴量 + 画像
- ラベルごとのサンプル数: 10
- 分類クラス数: 99
- 訓練データ数: 990(99x10)
- 検証データ数: 594

## Solutions
- 画像と特徴量を入力，出力を99クラスのソフトマックスとするネットワークを作る．
- 画像数が少ないので小さめのシンプルなネットワークにする．

## Memo
- 画像のみや数値特徴量のみだとあんまり性能がよくない．
- 自作のImageDataGeneratorの作り方を学ぶことができる．
- 入力サイズは300x300x3の方が224x224x3より性能が良かった(224x224x3だと0.09どまり)
- 画像と数値的特徴量の混合ネットワークを構築することで分類性能を向上させる方法を学べる．

## References
- \[1\]. [Kaggle - Leaf Classification](https://www.kaggle.com/c/leaf-classification)
- \[2\]. [Kaggle - Keras ConvNet LB 0.0052 w/ Visualization](https://www.kaggle.com/abhmul/leaf-classification/keras-convnet-lb-0-0052-w-visualization)
- \[3\]. [Keras - Functional API Guide](https://keras.io/ja/getting-started/functional-api-guide/)
