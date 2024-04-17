import torch
import numpy as np


class Model:
    def __init__(self, x_train, y_train, nodes_1=32, nodes_2=16, nodes_3=4):
        self.x = torch.tensor(x_train, dtype=torch.float)
        self.y = torch.tensor(y_train, dtype=torch.float)
        self.row_x, self.line_x = x_train.shape
        self.layer_1 = torch.randn((self.line_x, nodes_1), dtype=torch.float, requires_grad=True)
        self.b_1 = torch.randn(nodes_1, dtype=torch.float, requires_grad=True)

        self.layer_2 = torch.randn((nodes_1, nodes_2), dtype=torch.float, requires_grad=True)
        self.b_2 = torch.randn(nodes_2, dtype=torch.float, requires_grad=True)

        self.layer_3 = torch.randn((nodes_2, nodes_3), dtype=torch.float, requires_grad=True)
        self.b_3 = torch.randn(nodes_3, dtype=torch.float, requires_grad=True)

        self.output = torch.randn((nodes_3, 1), dtype=torch.float, requires_grad=True)
        self.b_y = torch.randn(1, dtype=torch.float, requires_grad=True)

        self.losses = []

    def fit(self, alpha=0.01, steps=100):
        x = self.x
        y = self.y
        for i in range(steps):
            hidden_1 = torch.tanh(torch.matmul(x, self.layer_1) + self.b_1)
            hidden_2 = torch.tanh(torch.matmul(hidden_1, self.layer_2) + self.b_2)
            hidden_3 = torch.tanh(torch.matmul(hidden_2, self.layer_3) + self.b_3)
            y_predict = torch.tanh(torch.matmul(hidden_3, self.output) + self.b_y)

            loss = torch.var(y - y_predict)
            self.losses.append(loss.data.numpy())

            loss.backward()

            self.layer_1.data -= alpha * self.layer_1.grad
            self.b_1.data -= alpha * self.b_1.grad
            self.layer_2.data -= alpha * self.layer_2.grad
            self.b_2.data -= alpha * self.b_2.grad
            self.layer_3.data -= alpha * self.layer_3.grad
            self.b_3.data -= alpha * self.b_3.grad
            self.output.data -= alpha * self.output.grad
            self.b_y.data -= alpha * self.b_y.grad

            self.layer_1.grad.zero_()
            self.b_1.grad.zero_()
            self.layer_2.grad.zero_()
            self.b_2.grad.zero_()
            self.layer_3.grad.zero_()
            self.b_3.grad.zero_()
            self.output.grad.zero_()
            self.b_y.grad.zero_()

    def predict(self, x_predict):
        x = torch.tensor(x_predict,dtype=torch.float)
        hidden_1 = torch.tanh(torch.matmul(x, self.layer_1) + self.b_1)
        hidden_2 = torch.tanh(torch.matmul(hidden_1, self.layer_2) + self.b_2)
        hidden_3 = torch.tanh(torch.matmul(hidden_2, self.layer_3) + self.b_3)
        y_predict = torch.tanh(torch.matmul(hidden_3, self.output) + self.b_y)
        print(y_predict)


if __name__ == '__main__':
    # x_true = np.array(
    #     [[175, 75, 70],
    #      [170, 80, 85],
    #      [-160, -50, -10],
    #      [-165, -55, -15],
    #      [-170, -57, -10],
    #      [185, 87, 75],
    #      [188, 78, 90],
    #      [-155, -48, -5]]
    #     )
    # y_true = np.array(
    #     [1,
    #      1,
    #      -1,
    #      -1,
    #      -1,
    #      1,
    #      1,
    #      -1]
    # )
    # # x_predict = np.array([172, 77, 50])
    #
    # x_predict = np.array(
    #     [[172, 77, 50],
    #      [166, 70, 60],
    #      [-153, -49, -25],
    #      [-158, -52, -30]]
    # )

    x_true = np.array(
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
    y_true = np.array(
        [1,
         -1,
         -1,
         1,
         1,
         1,
         -1,
         -1,
         -1])
    x_predict = np.array(
        [[171.8, 62.01],
         [-169.2, -61.28],
         [-189.05, -47.1],
         [177.8, 86.2, ],
         [181, 96],
         [178.4, 72.05],
         [-171.36, -72.8],
         [-161.45, -62],
         [-168.76, -62]]
    )
    model = Model(x_true, y_true)
    model.fit()
    model.predict(x_predict)
