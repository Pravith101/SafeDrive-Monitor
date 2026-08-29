import torch
import torch.nn as nn

class TemporalDrowsinessGRU(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, num_classes=3):
        super(TemporalDrowsinessGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.3
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        gru_out, _ = self.gru(x, h0)
        final_timestep_out = gru_out[:, -1, :]
        return self.classifier(final_timestep_out)