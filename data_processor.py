import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    features = ['international_oil_price', 'exchange_rate', 'domestic_oil_price']

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])

    return scaled_data, scaler

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length, 2]  # domestic_oil_price를 타겟으로 설정
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)
