import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    [원리 설명]
    - LSTMModel 클래스는 LSTM (Long Short-Term Memory) 신경망 구조를 정의합니다.
    - LSTM은 시계열 데이터의 장기 의존성을 학습하기 위해 고안되었으며, 
      과거의 정보를 은닉 상태에 저장하여 현재 입력과 결합합니다.
    
    주요 구성 요소:
      1. LSTM 레이어:
         - 입력 시퀀스를 처리하며, 각 시점의 은닉 상태를 업데이트합니다.
         - 입력 벡터의 각 성분(예: 'Dubai_Val', 'KRW_Rating' 등)에 대해 학습 과정에서 가중치가 조정됩니다.
         - 이 가중치들은 학습 데이터에 의해 최적화되며, 사전에 고정된 값은 없습니다.
      2. Fully Connected (Linear) Layer:
         - LSTM의 마지막 시점의 은닉 상태를 입력받아 최종 출력(타겟 예측)을 계산합니다.
         - 이 레이어의 가중치도 학습 과정에서 결정됩니다.
    
    파라미터:
      - input_dim: 입력 특성의 수 (예: 여러 경제 및 원유 지표의 총 개수)
      - hidden_dim: LSTM 내부 은닉 상태의 차원 (학습 중 내부 표현 벡터의 크기)
      - layer_dim: LSTM 레이어의 수
      - output_dim: 출력 특성의 수 (예: 타겟 유가 변수 4개)
    """
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        [원리 설명]
        - 입력 x에 대해 초기 은닉 상태(h0)와 셀 상태(c0)를 0으로 초기화합니다.
        - LSTM 레이어를 통해 입력 시퀀스의 정보를 처리하고, 마지막 시점의 은닉 상태를 추출합니다.
        - 이 은닉 상태를 Fully Connected Layer에 전달하여 최종 예측값(타겟 변수)을 계산합니다.
        
        반환:
          - 최종 출력: 타겟 변수의 예측값 (예: 4개 유가 변수)
        """
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
