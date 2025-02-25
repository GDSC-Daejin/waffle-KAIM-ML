import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def normalize_data(data):
    """
    [원리 설명]
    - 입력 데이터의 각 컬럼을 0과 1 사이로 정규화합니다.
    - MinMaxScaler를 사용해 각 열의 최소/최대값을 기준으로 선형 변환을 수행합니다.
    
    파라미터:
      - data: 정규화할 numpy 배열
    반환:
      - 정규화된 데이터와 이후 역정규화에 사용할 scaler 객체
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data), scaler

def create_dataset(dataset, look_back=1):
    """
    [원리 설명]
    - 시계열 데이터를 학습용 데이터셋으로 변환하는 함수입니다.
    - 연속된 look_back 기간의 데이터를 하나의 입력(X) 샘플로 구성하고,
      그 바로 다음 시점의 데이터를 타겟(Y)으로 설정합니다.
      
    예:
      - look_back=3이면, 인덱스 0~2를 X, 3번째 데이터를 Y로 사용합니다.
      - 이후 1~3을 X, 4번째를 Y로 하는 식으로 슬라이딩 윈도우를 구성합니다.
    
    파라미터:
      - dataset: 정규화된 전체 데이터 (numpy 배열)
      - look_back: 입력 시퀀스 길이 (일수)
    반환:
      - X: (샘플수, look_back, 특성 수) 배열
      - Y: (샘플수, 특성 수) 배열
    """
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back])
        Y.append(dataset[i+look_back])
    return np.array(X), np.array(Y)

def prepare_data(X, Y):
    """
    [원리 설명]
    - numpy 배열 형태의 데이터를 PyTorch 텐서로 변환합니다.
    - LSTM 모델 학습에 사용하기 위해 torch.FloatTensor 형태로 바꿉니다.
    
    파라미터:
      - X: 입력 데이터 (numpy 배열)
      - Y: 타겟 데이터 (numpy 배열)
    반환:
      - (X, Y) 형태의 PyTorch 텐서
    """
    return torch.FloatTensor(X), torch.FloatTensor(Y)
