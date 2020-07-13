import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        #input size 그대로 유지함
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=9, padding=4)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=9, padding=4)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=7, padding=3)
        self.relu = nn.ReLU()

    def forward(self, img):
        out = self.relu(self.conv1(img))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.conv4(out)

        return out