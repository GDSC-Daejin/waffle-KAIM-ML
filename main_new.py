import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터 로드 및 전처리
file_path = "korea_economic_data.csv"  # 파일 경로 설정
df = pd.read_csv(file_path)

# 날짜 변환 및 정렬
df['date'] = pd.to_datetime(df['date'].str.replace('Date_', '', regex=False), format='%Y_%m_%d')
df = df.sort_values(by='date').reset_index(drop=True)

# 전국 평균 유가 추출
df['gasoline'] = df['gasoline'].apply(lambda x: float(eval(x)[0]))

# 정규화 (0~1 스케일)
scaler = MinMaxScaler()
df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

# 2. LSTM 입력 데이터 생성 함수
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # 전국 평균 유가 (gasoline)
    return np.array(X), np.array(y)

# 시퀀스 길이 설정
seq_length = 30
data_values = df.iloc[:, 1:].values
X, y = create_sequences(data_values, seq_length)

# 80% 훈련 데이터, 20% 테스트 데이터로 분할
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# PyTorch 텐서 변환 -- 주의
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# 3. LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# 모델 하이퍼파라미터 설정
input_size = X_train.shape[2]
hidden_size = 50
num_layers = 2
epochs = 300
learning_rate = 0.001

# 모델 초기화
model = LSTMModel(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 4️ 모델 학습 함수
def train_model(model, X_train, y_train, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 5️ 모델 학습 실행
train_model(model, X_train, y_train, criterion, optimizer, epochs)

# 6️ 모델 평가 함수 (RMSE 계산) -- *확인필요*
def evaluate_model(model, X_test, y_test, criterion):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mse = criterion(predictions.squeeze(), y_test)
        rmse = torch.sqrt(mse)
    return rmse.item()

# 7️ 테스트 데이터 평가
rmse = evaluate_model(model, X_test, y_test, criterion)
print(f'Test RMSE: {rmse:.4f}')

#8️ 일주일치 예측 (7일 예측)
def predict_future(model, data, seq_length, days=7):
    model.eval()
    predictions = []
    last_sequence = data[-seq_length:].tolist()  # 마지막 30일 데이터

    with torch.no_grad():
        for _ in range(days):
            input_seq = torch.tensor([last_sequence], dtype=torch.float32)  # (1, 30, features)
            pred = model(input_seq).item()
            predictions.append(pred)
            
            # 새로운 값을 시퀀스에 추가하고 첫 번째 값 제거 (Sliding Window 방식)
            last_sequence.pop(0)
            last_sequence.append([pred] + [0] * (len(last_sequence[0]) - 1))  # 첫 번째 컬럼(유가)만 업데이트

    return predictions

# 9️ 예측 실행 및 결과 출력
future_predictions = predict_future(model, data_values, seq_length, days=7)

# 10️ 정규화 복원 (실제 유가 값 변환)
future_predictions_real = scaler.inverse_transform([[p] + [0] * (data_values.shape[1] - 1) for p in future_predictions])

#  예측된 유가 출력 (향후 7일)
print("\n 전국 평균 유가 예측 (향후 7일):")
for i, price in enumerate(future_predictions_real, 1):
    print(f"Day {i}: {price[0]:.2f} 원")

# 11️ 예측 결과 시각화
days = range(1, 8)
plt.figure(figsize=(8, 5))
plt.plot(days, [p[0] for p in future_predictions_real], marker='o', linestyle='--', label='Predicted Price')
plt.xlabel('Days')
plt.ylabel('Gasoline Price (KRW)')
plt.title('7-Day Gasoline Price Forecast')
plt.legend()
plt.grid()
plt.show()

from sklearn.preprocessing import MinMaxScaler

# 지역명 컬럼 제외 (숫자형 데이터만 선택)
numeric_cols = df.select_dtypes(include=["number"]).columns  # 숫자 데이터만 선택
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("* 데이터 스케일링 완료!")


