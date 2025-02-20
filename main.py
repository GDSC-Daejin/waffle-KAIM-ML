from data_processor import load_and_preprocess_data, create_sequences
from model import OilPriceLSTM
from train import train_model, evaluate_model
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

def main():
    # 데이터 로드 및 전처리
    data, scaler = load_and_preprocess_data('oil_price_data.csv')
    X, y = create_sequences(data, seq_length=30)

    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # PyTorch 텐서로 변환
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).view(-1, 1)

    # 모델 파라미터 설정
    input_size = X_train.shape[2]
    hidden_size = 64
    num_layers = 2
    output_size = 1

    # 모델 초기화
    model = OilPriceLSTM(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 모델 훈련
    epochs = 1000
    train_model(model, X_train, y_train, criterion, optimizer, epochs)

    # 모델 평가
    rmse = evaluate_model(model, X_test, y_test, criterion)
    print(f'Test RMSE: {rmse:.4f}')

    # 예측 수행
    model.eval()
    with torch.no_grad():
        last_sequence = X_test[-1].unsqueeze(0)
        prediction = model(last_sequence)
        unscaled_prediction = scaler.inverse_transform(prediction.numpy())
        print(f'Next day prediction: {unscaled_prediction[0][0]:.2f}')

if __name__ == "__main__":
    main()
