import torch
import torch.nn as nn

class TabPFNMechanism(nn.Module):
    """
    A PyTorch module implementing a small MLP mechanism.
    """
    def __init__(self, input_dim, hidden_dim=16):
        super(TabPFNMechanism, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        for p in self.net.parameters():
            nn.init.normal_(p, mean=0, std=0.1)

    def forward(self, x):
        with torch.no_grad():
            return self.net(torch.FloatTensor(x)).numpy().squeeze()
