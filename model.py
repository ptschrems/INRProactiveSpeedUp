import torch
import torch.nn as nn

class SirenBlock(nn.Module):
    def __init__(self, in_features, out_features, omega=30.0, is_first=False):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features)
        
        # Proper SIREN weight initialization (Sitzmann et al.)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                self.linear.weight.uniform_(
                    -torch.sqrt(torch.tensor(6.0 / in_features)) / omega,
                     torch.sqrt(torch.tensor(6.0 / in_features)) / omega
                )

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class Siren(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=256, out_dim=3, num_layers=4, omega_0=30.0):
        """
        in_dim: 2 for (x, y) coordinates
        out_dim: 3 for RGB values
        """
        super().__init__()
        self.layers = nn.ModuleList()
        
        # First layer (omega_0 = 30.0)
        self.layers.append(SirenBlock(in_dim, hidden_dim, omega=omega_0, is_first=True))
        
        # Hidden layers (omega = 1.0)
        for _ in range(num_layers - 2):
            self.layers.append(SirenBlock(hidden_dim, hidden_dim, omega=1.0))
            
        # Final linear output layer (maps 256 features -> 3 RGB channels)
        self.final_layer = nn.Linear(hidden_dim, out_dim)
        with torch.no_grad():
            self.final_layer.weight.uniform_(
                -torch.sqrt(torch.tensor(6.0 / hidden_dim)) / 30.0,
                 torch.sqrt(torch.tensor(6.0 / hidden_dim)) / 30.0
            )

    def forward(self, coords):
        # coords shape: (H, W, 2) or (N, 2)
        x = coords
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)