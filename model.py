import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores * 5)

        pruned_weights = self.weight * gates

        return F.linear(x, pruned_weights, self.bias)


class PrunableNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = PrunableLinear(32*32*3, 1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.fc3 = PrunableLinear(512, 256)
        self.fc4 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.3)

        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.3)

        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

    def get_all_gates(self):
        gates = []

        for layer in self.modules():
            if isinstance(layer, PrunableLinear):
                gates.append(torch.sigmoid(layer.gate_scores).view(-1))

        return torch.cat(gates)