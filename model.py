import torch
import torch.nn as nn
import numpy as np  # 수정된 부분

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
        self.bidirectional = True  # 양방향 사용 여부 명시적으로 저장
        
        # LSTM 계층
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            layer_dim, 
            batch_first=True, 
            dropout=dropout if layer_dim > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # 양방향이면 hidden_dim * 2
        fc_input_dim = hidden_dim * 2 if self.bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 입력 차원 확인 및 조정
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        # 배치 크기 가져오기
        batch_size = x.size(0)
        
        # 중요: bidirectional이면 layer_dim * 2 사용
        num_directions = 2 if self.bidirectional else 1
        device = x.device
        
        # 올바른 크기의 hidden state 초기화
        h0 = torch.zeros(self.layer_dim * num_directions, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.layer_dim * num_directions, batch_size, self.hidden_dim, device=device)
        
        # LSTM 실행
        out, _ = self.lstm(x, (h0, c0))
        
        # 마지막 시퀀스 출력 사용
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        out = self.relu(out)
        out = self.output_layer(out)
        
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

# 앙상블 모델 클래스 추가
class EnsembleModel(BaseModel):
    def __init__(self, models, weights=None):
        super(EnsembleModel, self).__init__()
        self.models = models
        
        # 가중치가 제공되지 않으면 모든 모델에 동일한 가중치 할당
        if weights is None:
            self.weights = torch.ones(len(models)) / len(models)
        else:
            # 가중치 정규화
            total_weight = sum(weights)
            self.weights = torch.tensor([w/total_weight for w in weights])
    
    def to(self, device):
        # 모든 하위 모델을 지정된 디바이스로 이동
        for i in range(len(self.models)):
            self.models[i] = self.models[i].to(device)
        return self
    
    def forward(self, x):
        # 각 모델의 출력을 가중치와 함께 평균
        outputs = []
        device = x.device
        
        for i, model in enumerate(self.models):
            model_output = model(x)
            if isinstance(model_output, torch.Tensor):
                outputs.append(model_output * self.weights[i])
            else:
                outputs.append(torch.tensor(model_output, device=device) * self.weights[i])
        
        # 모델 출력 합산
        ensemble_output = sum(outputs)
        return ensemble_output
    
    def predict(self, x):
        # 추론 모드로 전환
        for model in self.models:
            model.eval()
        
        # 그래디언트 계산 비활성화
        with torch.no_grad():
            output = self.forward(x)
            
            # 텐서를 NumPy 배열로 변환
            if isinstance(output, torch.Tensor):
                return output.cpu().numpy()
            else:
                return output

def create_model(model_type, input_dim, hidden_dim, output_dim, **kwargs):
    """
    모델 타입에 따라 적절한 모델을 생성하는 함수
    
    Args:
        model_type: 모델 타입 ('lstm', 'gru', 'cnn', 'transformer')
        input_dim: 입력 차원
        hidden_dim: 은닉 차원
        output_dim: 출력 차원
        **kwargs: 추가 파라미터
        
    Returns:
        생성된 모델 인스턴스
    """
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
