import torch
import torch.nn as nn
import torch.autograd as autograd

class Classifier(nn.Module):
    def __init__(self, state_shape, num_states):
        super(Classifier, self).__init__()

        self.state_shape = state_shape
        self.features = nn.Sequential(
                nn.Conv2d(self.state_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                )
        self.fc = nn.Sequential(
                nn.Linear(self.feature_size(), 512),
                nn.ReLU(),
                )
        self.head = nn.Linear(512, num_states)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        loc = self.head(x)
        return loc

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.state_shape))).view(1, -1).size(1)
