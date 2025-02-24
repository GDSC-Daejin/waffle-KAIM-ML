import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def normalize_data(data):
    """데이터를 [0, 1] 범위로 정규화합니다."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data), scaler

def create_dataset(dataset, look_back=1):
    """
    시계열 데이터셋을 생성합니다.
    이전 look_back일의 데이터를 이용해 다음 날 데이터를 예측할 수 있도록 입력(X)과 출력(Y)를 만듭니다.
    """
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back])
        Y.append(dataset[i+look_back])
    return np.array(X), np.array(Y)

def prepare_data(X, Y):
    """numpy 배열을 PyTorch 텐서로 변환합니다."""
    return torch.FloatTensor(X), torch.FloatTensor(Y)
