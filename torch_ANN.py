import torch
import torch.nn as nn
import numpy as np


class Model:
    def __init__(self, x_train, y_train, nodes_layer_1=5120):
        self.row_x, self.line_x = x_train.shape
        self.layer_1 = torch.tensor((self.line_x, nodes_layer_1),dtype=torch.float)
        self.b_1 = torch.tensor(nodes_layer_1,dtype=torch.float)

        self.layer_2 = torch.tensor((nodes_layer_1, 1),dtype=torch.float)
        self.b_2 = torch.tensor(1,dtype=torch.float)


        print(self.row_x,self.line_x,sep=',')

    def fit(self, alpha=0.01, steps=1000):
        pass

    def predict(self, x_predict):
        pass


if __name__ == '__main__':
    data = np.array(
        [[175, 77],
         [-161, -65],
         [-159, -43],
         [175, 92],
         [185, 106],
         [171, 82],
         [-159, -90],
         [-167.8, -78],
         [-150.76, -75]
         ])
    y = np.array(
        [1,
         0,
         0,
         1,
         1,
         1,
         0,
         0,
         0])
    x_test = np.array(
        [[171.8, 62.01],
         [-169.2, -61.28],
         [-189.05, -47.1],
         [177.8, 86.2,],
         [181, 96],
         [178.4, 72.05],
         [-171.36, -72.8],
         [-161.45, -62],
         [-168.76, -62]]
    )

    model = Model(data, y)
    model.fit()
    model.predict(x_test)

