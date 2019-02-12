from torch.nn import Module, Sequential, Conv2d, ReLU

class SRCNN(Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.net = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4),
            ReLU(),
            Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0),
            ReLU(),
            Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)
        )

    def forward(self, x):
        x = self.net(x)
        return x