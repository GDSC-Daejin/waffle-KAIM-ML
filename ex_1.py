import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# LSTM 모델 정의
class OilPriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(OilPriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 데이터 로드 및 전처리
def load_and_preprocess_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    selected_features = data[:, 1:4]  # 국제유가, 환율, 국내유가 사용
    min_vals = selected_features.min(axis=0)
    max_vals = selected_features.max(axis=0)
    scaled_data = (selected_features - min_vals) / (max_vals - min_vals)
    return scaled_data, min_vals, max_vals

# 시퀀스 데이터 생성
def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length, -1]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# 모델 학습 함수
def train_model(model, X_train, y_train, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 모델 평가 함수
def evaluate_model(model, X_test, y_test, criterion):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mse = criterion(predictions, y_test)
        rmse = torch.sqrt(mse)
    return rmse.item()

# 메인 실행 함수
def main():
    data, min_vals, max_vals = load_and_preprocess_data('oil_price_data.csv')
    X, y = create_sequences(data, seq_length=30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).view(-1, 1)
    
    input_size = X_train.shape[2]
    model = OilPriceLSTM(input_size, hidden_size=64, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, X_train, y_train, criterion, optimizer, epochs=1000)
    rmse = evaluate_model(model, X_test, y_test, criterion)
    print(f'Test RMSE: {rmse:.4f}')
    
    model.eval()
    with torch.no_grad():
        last_sequence = X_test[-1].unsqueeze(0)
        prediction = model(last_sequence)
        unscaled_prediction = prediction.numpy() * (max_vals[-1] - min_vals[-1]) + min_vals[-1]
        print(f'Next day prediction: {unscaled_prediction[0][0]:.2f}')

if __name__ == "__main__":
    main()
