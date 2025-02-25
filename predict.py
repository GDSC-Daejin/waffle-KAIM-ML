import torch
import numpy as np

def make_predictions(model, X):
    """
    학습된 LSTM 모델로부터 예측값을 생성하는 함수입니다.
    """
    model.eval()
    with torch.no_grad():
        return model(X).numpy()

def calculate_rmse(y_true, y_pred):
    """
    RMSE(평균제곱근오차)를 계산하는 함수입니다.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def predict_future(model, last_sequence, future_steps, scaler):
    """
    마지막 look_back 기간의 데이터를 사용하여 future_steps만큼 미래를 예측합니다.
    :param model: 학습된 LSTM 모델
    :param last_sequence: 마지막 look_back일의 정규화된 데이터 (shape: (look_back, num_features))
    :param future_steps: 예측할 미래 일 수
    :param scaler: 정규화에 사용된 MinMaxScaler (역정규화에 사용)
    :return: 역정규화된 예측 결과, shape: (future_steps, num_features)
    """
    model.eval()
    predictions = []
    current_seq = last_sequence.copy()
    for _ in range(future_steps):
        input_tensor = torch.FloatTensor(current_seq).unsqueeze(0)
        with torch.no_grad():
            pred = model(input_tensor).numpy()[0]
        predictions.append(pred)
        # 예측값을 시퀀스에 추가 (슬라이딩)
        current_seq = np.vstack([current_seq[1:], pred])
    predictions = np.array(predictions)
    predictions = scaler.inverse_transform(predictions)
    return predictions
