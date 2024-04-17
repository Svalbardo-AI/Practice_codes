import torch
import numpy as np


class Model:
    def __init__(self, x_train, y_train, nodes_layer_1=5120):
        self.x = torch.tensor(x_train,dtype=torch.float)
        self.y = torch.tensor(y_train,dtype=torch.float)
        self.row_x, self.line_x = x_train.shape
        self.layer_1 = torch.randn((self.line_x, nodes_layer_1),dtype=torch.float, requires_grad=True)
        self.b_1 = torch.randn(nodes_layer_1,dtype=torch.float, requires_grad=True)

        self.layer_2 = torch.randn((nodes_layer_1, 1),dtype=torch.float, requires_grad=True)
        self.b_2 = torch.randn(1,dtype=torch.float, requires_grad=True)

        self.losses = []

    def fit(self, alpha=0.01, steps=10):
        x = self.x
        y = self.y

        for i in range(steps):
            y_hidden = torch.tanh(torch.matmul(x,self.layer_1)+self.b_1)
            y_predict = torch.tanh(torch.matmul(y_hidden,self.layer_2)+self.b_2)
            loss = torch.var(y-y_predict)
            self.losses.append(loss.data.numpy())

            loss.backward()

            self.layer_1.data -= alpha * self.layer_1.grad
            self.b_1.data -= alpha * self.b_1.grad
            self.layer_2.data -= alpha * self.layer_2.grad
            self.b_2.data -= alpha * self.b_2.grad

            self.layer_1.grad.data.zero_()
            self.b_1.grad.data.zero_()
            self.layer_2.grad.data.zero_()
            self.b_2.grad.data.zero_()

    def predict(self, x_predict):
        x = torch.tensor(x_predict,dtype=torch.float)

        y_hidden = torch.tanh(torch.matmul(x,self.layer_1)+self.b_1)
        y_predict = torch.tanh(torch.matmul(y_hidden,self.layer_2)+self.b_2)

        print(y_predict)


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
         -1,
         -1,
         1,
         1,
         1,
         -1,
         -1,
         -1])
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

