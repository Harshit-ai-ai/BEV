import torch.nn as nn

class OccupancyHead(nn.Module):
    def __init__(self):
        super(OccupancyHead, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(x)