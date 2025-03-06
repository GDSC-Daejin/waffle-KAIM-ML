import torch
import torch.nn as nn
import numpy as np

# Base model class for shared methods
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError

# LSTM model
class LSTMModel(BaseModel):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout if layer_dim > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# GRU model
class GRUModel(BaseModel):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout if layer_dim > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# CNN model
class CNNModel(BaseModel):
    def __init__(self, input_dim, hidden_dim, output_dim, sequence_length, dropout=0.2):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(hidden_dim * (sequence_length // 2), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.conv1(x)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Transformer model
class TransformerModel(BaseModel):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(hidden_dim, nhead, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer(x)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out

def create_model(model_type, input_dim, hidden_dim, output_dim, **kwargs):
    if model_type == 'lstm':
        return LSTMModel(input_dim, hidden_dim, kwargs.get('layer_dim', 1), output_dim, kwargs.get('dropout', 0.2))
    elif model_type == 'gru':
        return GRUModel(input_dim, hidden_dim, kwargs.get('layer_dim', 1), output_dim, kwargs.get('dropout', 0.2))
    elif model_type == 'cnn':
        sequence_length = kwargs.get('sequence_length', 3)
        return CNNModel(input_dim, hidden_dim, output_dim, sequence_length, kwargs.get('dropout', 0.2))
    elif model_type == 'transformer':
        nhead = kwargs.get('nhead', 4)
        num_layers = kwargs.get('num_layers', 2)
        return TransformerModel(input_dim, hidden_dim, output_dim, nhead, num_layers, kwargs.get('dropout', 0.2))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
