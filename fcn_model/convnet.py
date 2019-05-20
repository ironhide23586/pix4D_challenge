import torch.nn as nn


class FCN(nn.Module):

    def __init__(self):
        super(FCN, self).__init__()
        self.layers = []
        self.layers.append(nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ))
        self.layers.append(nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ))
        self.layers.append(nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ))
        self.layers.append(nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=1),
            nn.Conv2d(8, 2, kernel_size=5, stride=1, padding=2),
            # nn.Softmax2d()
        ))

        self.all_train_layers = nn.Sequential(
            self.layers[0],
            self.layers[1],
            self.layers[2],
            self.layers[3]
        )

        self.softmax_layer = nn.Softmax2d()

    def forward(self, x):
        out = self.all_train_layers(x)
        return out

    def infer_softmax(self, x):
        out = self.all_train_layers(x)
        out = self.softmax_layer(out)
        return out