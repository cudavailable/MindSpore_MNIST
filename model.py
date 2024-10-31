import mindspore
from mindspore import nn

# define a model
class Net(nn.Cell):
      def __init__(self):
            super().__init__()
            # input : n*1*28*28
            self.conv1 = nn.SequentialCell(
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, pad_mode='valid'), # n*6*24*24
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), # n*6*12*12
            )

            self.conv2 = nn.SequentialCell(
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, pad_mode='valid'), # n*16*8*8
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), # n*16*4*4
            )

            self.flatten = nn.Flatten() # n * 256

            self.dense_relu_sequential = nn.SequentialCell(
                  nn.Dense(256, 120), # Dense = Linear (Layer)
                  nn.ReLU(),
                  nn.Dense(120, 84),
                  nn.ReLU(),
                  nn.Dense(84, 10)
            )

      def construct(self, x):
            x = self.conv1(x) # convolutional layers
            x = self.conv2(x)
            x = self.flatten(x)
            logits = self.dense_relu_sequential(x)
            return logits