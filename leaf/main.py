from keras.layers.core import Flatten
from keras.layers import Input, Activation, Dense, merge, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import random_zoom, flip_axis
from keras.models import Model
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import cv2

NB_CLASSES = 99

def _conv_bn_relu(x, nb_filters):
    x = Convolution2D(32, 3, 3, border_mode="same", init="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x 

def LeafNet():
    """
    References
    ----------
    ..[1]. Kaggle - Leaf Classification, <https://www.kaggle.com/c/leaf-classification>
    ..[2]. AbhijeetMulgund, Kaggle - Keras ConvNet LB 0.0052 w/ Visualization, <https://www.kaggle.com/abhmul/leaf-classification/keras-convnet-lb-0-0052-w-visualization>
    ..[3]. Keras Documentation - Getting started with the Keras functional API, <https://keras.io/ja/getting-started/functional-api-guide/>
    ..[4]. Keras Issues - A concrete example for using data generator for large datasets such as ImageNet #1627, <https://github.com/fchollet/keras/issues/1627>
    """
    # Image Network
    image_input = Input(shape=(96, 96, 1))
    x = _conv_bn_relu(image_input, 32)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = _conv_bn_relu(x, 64)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = _conv_bn_relu(x, 128)
    x = Flatten()(x)
    x = Dense(96, activation="relu")(x)
    # Numerical Network
    numeric_input = Input(shape=(192,))
    # Combined Network 
    z = merge([x, numeric_input], mode="concat")
    z = Dense(NB_CLASSES, activation="softmax")(z)
    model = Model(input=[image_input, numeric_input], output=z)
    print model.summary()
    return model

def load_train():
    # id,species,margin1,margin2,...
    train_df = pd.read_csv("../input/train.csv")
    label_index_mapping = {label: index for index, label in enumerate(sorted(train_df["species"].unique()))}
    ID = train_df["id"].values
    y = train_df["species"].map(label_index_mapping).values
    del train_df["id"], train_df["species"]
    X_image = np.array([cv2.resize(cv2.imread("../input/images/%d.jpg" % id, 0), (96, 96)) for id in ID])
    X_image = X_image.reshape((len(X_image), 96, 96, 1))
    X_image = X_image / 255.0
    X_numeric = train_df.values
    return X_image, X_numeric, y, ID

def flow(X, Z, Y):
    N = len(X)
    batch_size = 32
    while True:
        for i in range(0, N, batch_size):
            X_i = X[i:i+batch_size].copy()
            for j in range(len(X_i)):
                if np.random.random() < 0.5:
                    X_i[j] = flip_axis(X_i[j], 0)
                if np.random.random() < 0.5:
                    X_i[j] = flip_axis(X_i[j], 1)
                X_i[j] = random_zoom(X_i[j], (0.8, 1.2))
            yield [X_i, Z[i:i+batch_size]], Y[i:i+batch_size]

def train_test_split(X, kfold_index):
    kf = KFold(n_splits=5, shuffle=True, random_state=7777)
    return list(kf.split(X))[kfold_index]

def run():
    model = LeafNet()
    X, Z, y, ID = load_train()
    Y = to_categorical(y, NB_CLASSES) 
    train_index, test_index = train_test_split(ID, 0)
    X_train, X_test = X[train_index], X[test_index]
    Z_train, Z_test = Z[train_index], Z[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    model.fit_generator(
        generator=flow(X_train, Z_train, Y_train), 
        samples_per_epoch=len(X),
        nb_epoch=100,
        validation_data=([X_test, Z_test], Y_test)
        ) 

if __name__ == "__main__":
    run()
