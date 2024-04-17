import torch
import numpy as np


class Model:
    def __init__(self, x_train, y_train, nodes_1=5120, nodes_2=640, nodes_3=80):
        pass

    def fit(self, alpha=0.01, steps=1000):
        pass

    def predict(self, y_predict):
        pass


if __name__ == '__main__':
    x = []
    y=[]
    x_predict=[]
    model = Model(x,y)
    model.fit()
    model.predict(x_predict)
