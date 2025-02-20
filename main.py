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
        
        # LSTM 레이어
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully Connected 레이어 (출력 레이어)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 초기 hidden state와 cell state를 0으로 설정
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM을 통해 순전파 진행
        out, _ = self.lstm(x, (h0, c0))
        
        # 마지막 타임스텝의 출력을 FC 레이어를 통해 최종 예측값 산출
        out = self.fc(out[:, -1, :])
        return out

# 데이터 로드 및 전처리
def load_and_preprocess_data(file_path):
    # CSV 파일 로드 (NumPy 사용)
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

    # 국제유가, 환율, 국내유가 선택 (1, 2, 3번째 컬럼 사용)
    selected_features = data[:, 1:4]  

    # Min-Max 정규화 (0~1 범위로 변환)
    min_vals = selected_features.min(axis=0)
    max_vals = selected_features.max(axis=0)
    scaled_data = (selected_features - min_vals) / (max_vals - min_vals)
    
    return scaled_data, min_vals, max_vals

# 시퀀스 데이터 생성 (LSTM 모델 입력용)
def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]  # seq_length 만큼 연속된 데이터 가져오기
        target = data[i+seq_length, -1]  # 다음 시점의 국내유가를 예측값으로 설정
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)

# 모델 학습 함수
def train_model(model, X_train, y_train, criterion, optimizer, epochs):
    model.train()  # 학습 모드 활성화
    for epoch in range(epochs):
        outputs = model(X_train)  # 예측 수행
        loss = criterion(outputs, y_train)  # 손실 계산
        
        optimizer.zero_grad()  # 기울기 초기화
        loss.backward()  # 역전파 수행
        optimizer.step()  # 가중치 업데이트

        # 100 epoch마다 손실 출력
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 모델 평가 함수
def evaluate_model(model, X_test, y_test, criterion):
    model.eval()  # 평가 모드 활성화
    with torch.no_grad():  # 기울기 업데이트 방지
        predictions = model(X_test)  # 예측 수행
        mse = criterion(predictions, y_test)  # MSE 계산
        rmse = torch.sqrt(mse)  # RMSE 계산
    return rmse.item()

# 메인 실행 함수
def main():
    # 데이터 로드 및 전처리
    data, min_vals, max_vals = load_and_preprocess_data('oil_price_data.csv')
    
    # 시퀀스 데이터 생성
    X, y = create_sequences(data, seq_length=30)

    # 훈련/테스트 데이터 분할 (80% 훈련, 20% 테스트)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # NumPy 데이터를 PyTorch 텐서로 변환
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).view(-1, 1)

    # 모델 정의
    input_size = X_train.shape[2]  # 입력 특성 개수
    model = OilPriceLSTM(input_size, hidden_size=64, num_layers=2, output_size=1)

    # 손실 함수 및 옵티마이저 설정
    criterion = nn.MSELoss()  # 평균제곱오차 (MSE) 사용
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 옵티마이저 사용

    # 모델 학습
    train_model(model, X_train, y_train, criterion, optimizer, epochs=1000)

    # 모델 평가
    rmse = evaluate_model(model, X_test, y_test, criterion)
    print(f'Test RMSE: {rmse:.4f}')  # RMSE 출력

    # 예측 수행 (마지막 테스트 시퀀스를 이용)
    model.eval()
    with torch.no_grad():
        last_sequence = X_test[-1].unsqueeze(0)  # 마지막 시퀀스를 가져옴
        prediction = model(last_sequence)  # 예측 수행

        # 정규화 해제 (실제 가격으로 변환)
        unscaled_prediction = prediction.numpy() * (max_vals[-1] - min_vals[-1]) + min_vals[-1]
        print(f'Next day prediction: {unscaled_prediction[0][0]:.2f}')

# 프로그램 실행
if __name__ == "__main__":
    main()