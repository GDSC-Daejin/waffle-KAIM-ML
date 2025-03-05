import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """모델 구현을 위한 기본 추상 클래스"""
    
    @abstractmethod
    def forward(self, x):
        pass
    
    @property
    def name(self):
        return self.__class__.__name__

class LSTMModel(BaseModel):
    """
    LSTM 모델: 시계열 데이터에서 장기 의존성을 학습하는 모델
    이전 시점의 정보를 기억하는 메모리 셀을 통해 시간적 패턴을 학습합니다.
    """
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

class GRUModel(BaseModel):
    """
    GRU 모델: LSTM의 간소화된 변형
    LSTM보다 적은 매개변수로 비슷한 성능을 얻을 수 있으며 계산 효율이 높습니다.
    """
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

class CNNModel(BaseModel):
    """
    1D-CNN 모델: 시계열 데이터에서 국소적 패턴을 학습하는 모델
    컨볼루션 필터를 통해 시계열 데이터의 국소적 특성을 추출합니다.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, sequence_length, dropout=0.2):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim*2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        # 풀링 후 시퀀스 길이 계산
        pooled_length = sequence_length // 2
        self.fc = nn.Linear(hidden_dim * 2 * pooled_length, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # CNN은 [batch, channels, length] 형태 입력 필요
        x = x.permute(0, 2, 1)  # [batch, seq_len, features] -> [batch, features, seq_len]
        
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        
        # 플랫튼 하기 전에 형상 기억
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # 플랫튼
        x = self.dropout(x)
        x = self.fc(x)
        return x

class TransformerModel(BaseModel):
    """
    Transformer 모델: 자기 주의 메커니즘을 활용한 모델
    자기 주의 메커니즘으로 시계열 데이터의 장거리 의존성을 효과적으로 학습합니다.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, nhead=4, num_layers=2, dropout=0.2):
        super(TransformerModel, self).__init__()
        
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # batch_first=True 설정으로 경고 해결
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim*4, 
            dropout=dropout,
            batch_first=True  # 중요: 이 파라미터를 True로 설정
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        x = self.input_embedding(x)  # [batch, seq_len, hidden_dim]
        
        # batch_first=True 설정으로 더 이상 순서 변경이 필요 없음
        # x = x.permute(1, 0, 2)  # 이 줄 제거
        
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # 마지막 시퀀스의 출력 사용 - batch_first=True로 인한 인덱싱 변경
        x = x[:, -1]  # 이제 [batch, seq_len, hidden_dim]에서 마지막 시퀀스 선택
        x = self.dropout(x)
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    """Transformer 모델의 위치 인코딩"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # batch_first=True 설정에 맞게 위치 인코딩 형태 변경
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]  # 차원에 맞게 위치 인코딩 적용
        return self.dropout(x)

class EnsembleModel:
    """여러 모델의 예측을 결합하는 앙상블 모델"""
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights is not None else np.ones(len(models)) / len(models)
        
    def predict(self, x):
        """각 모델의 예측을 가중합하여 최종 예측 생성"""
        predictions = []
        
        # 입력 텐서 디바이스 확인
        device = x.device if isinstance(x, torch.Tensor) else torch.device('cpu')
        
        # 배치 크기 일관성 확인
        batch_size = x.shape[0] if isinstance(x, torch.Tensor) else x.shape[0]
        
        for model in self.models:
            # 모델 디바이스가 입력 텐서와 일치하는지 확인하고 필요시 이동
            model_device = next(model.parameters()).device
            if model_device != device:
                model = model.to(device)
                
            model.eval()
            with torch.no_grad():
                # 모델 예측 수행
                pred = model(x).cpu().numpy()
                
                # 예측 결과의 배치 크기 확인 및 수정
                if pred.shape[0] != batch_size:
                    # 모델이 배치 크기를 보존하지 않는 경우 (예: 첫 번째 샘플만 반환)
                    print(f"모델 출력 형태 조정: {pred.shape} -> ({batch_size}, {pred.shape[-1]})")
                    
                    # 대부분의 경우 모델이 (1, features) 형태의 출력을 생성하는데
                    # 이를 복제하여 원래 배치 크기로 확장
                    if pred.shape[0] == 1 and batch_size > 1:
                        pred = np.tile(pred, (batch_size, 1))
                    # 반대로 모델이 필요 이상의 샘플을 생성한 경우 자르기
                    elif pred.shape[0] > batch_size:
                        pred = pred[:batch_size]
                
                predictions.append(pred)
        
        # 모든 예측이 동일한 형태를 갖는지 확인
        shapes = [p.shape for p in predictions]
        if len(set(str(s) for s in shapes)) > 1:
            print(f"경고: 일관되지 않은 예측 형태: {shapes}")
            
            # 모든 예측을 첫 번째 예측 형태로 맞춤
            target_shape = predictions[0].shape
            for i in range(1, len(predictions)):
                if predictions[i].shape != target_shape:
                    # 형태가 다른 경우 재조정
                    if predictions[i].shape[0] == 1:
                        predictions[i] = np.tile(predictions[i], (target_shape[0], 1))
                    else:
                        # 크기가 다른 경우 첫 번째 샘플들만 사용
                        predictions[i] = predictions[i][:target_shape[0]]
        
        # 가중 평균 계산
        weighted_preds = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            weighted_preds += self.weights[i] * pred
            
        return weighted_preds
    
    # 모든 모델을 특정 디바이스로 이동하는 메서드 추가
    def to(self, device):
        for i, model in enumerate(self.models):
            self.models[i] = model.to(device)
        return self

def create_model(model_type, input_dim, hidden_dim, output_dim, **kwargs):
    """모델 타입에 따라 적절한 모델 인스턴스 생성"""
    if model_type == 'lstm':
        layer_dim = kwargs.get('layer_dim', 1)
        return LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, kwargs.get('dropout', 0.2))
    elif model_type == 'gru':
        layer_dim = kwargs.get('layer_dim', 1)
        return GRUModel(input_dim, hidden_dim, layer_dim, output_dim, kwargs.get('dropout', 0.2))
    elif model_type == 'cnn':
        sequence_length = kwargs.get('sequence_length', 3)
        return CNNModel(input_dim, hidden_dim, output_dim, sequence_length, kwargs.get('dropout', 0.2))
    elif model_type == 'transformer':
        nhead = kwargs.get('nhead', 4)
        num_layers = kwargs.get('num_layers', 2)
        return TransformerModel(input_dim, hidden_dim, output_dim, nhead, num_layers, kwargs.get('dropout', 0.2))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

