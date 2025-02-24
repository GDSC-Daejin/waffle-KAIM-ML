import torch
import numpy as np

def make_predictions(model, X):
    """
    학습된 모델로 예측값을 생성합니다.
    """
    model.eval()
    with torch.no_grad():
        return model(X).numpy()

def calculate_rmse(y_true, y_pred):
    """RMSE(평균제곱근오차)를 계산합니다."""
    return np.sqrt(np.mean((y_true - y_pred)**2))

def predict_future(model, last_sequence, future_steps, scaler):
    """
    마지막 look_back 데이터를 이용해 미래 future_steps일치 데이터를 예측합니다.
    :param model: 학습된 모델
    :param last_sequence: 마지막 look_back일의 정규화된 데이터 (shape: (look_back, num_features))
    :param future_steps: 예측할 미래 일 수 (예: 7)
    :param scaler: 정규화에 사용된 스케일러 (역정규화에 사용)
    :return: 미래 예측 결과 (역정규화 적용 후, shape: (future_steps, num_features))
    """
    model.eval()
    predictions = []
    current_seq = last_sequence.copy()
    for _ in range(future_steps):
        input_tensor = torch.FloatTensor(current_seq).unsqueeze(0)
        with torch.no_grad():
            pred = model(input_tensor).numpy()[0]
        predictions.append(pred)
        current_seq = np.vstack([current_seq[1:], pred])
    predictions = np.array(predictions)
    predictions = scaler.inverse_transform(predictions)
    return predictions
