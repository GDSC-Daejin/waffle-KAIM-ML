import torch
import numpy as np

def make_predictions(model, X):
    """
    [원리 설명]
    - 학습된 LSTM 모델을 사용하여 입력 데이터 X에 대한 예측값을 생성합니다.
    - torch.no_grad()를 사용하여 예측 시 그래디언트 계산을 생략함으로써 효율성을 높입니다.
    
    반환:
      - 예측 결과를 numpy 배열로 반환합니다.
    """
    model.eval()
    with torch.no_grad():
        return model(X).numpy()

def calculate_rmse(y_true, y_pred):
    """
    [원리 설명]
    - 실제 값(y_true)과 예측 값(y_pred) 사이의 평균제곱근오차(RMSE)를 계산합니다.
    - RMSE는 예측 오차의 크기를 나타내며, 값이 낮을수록 예측 성능이 우수함을 의미합니다.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def predict_future(model, last_sequence, future_steps, scaler):
    """
    [원리 설명]
    - 마지막 look_back 기간의 정규화된 데이터를 기반으로 미래 future_steps일치 예측값을 생성합니다.
    - 각 예측 후 슬라이딩 윈도우 방식을 적용하여, 입력 시퀀스를 갱신하고 연속적인 미래 값을 예측합니다.
    - 마지막에 scaler.inverse_transform()을 사용해, 정규화된 예측값을 원래 스케일로 복원합니다.
    
    파라미터:
      - model: 학습된 LSTM 모델
      - last_sequence: 마지막 look_back일의 정규화된 데이터 (shape: (look_back, num_features))
      - future_steps: 예측할 미래 일 수
      - scaler: MinMaxScaler 객체 (역정규화용)
      
    반환:
      - 미래 예측 결과 (원래 스케일, numpy 배열, shape: (future_steps, num_features))
    """
    model.eval()
    predictions = []
    current_seq = last_sequence.copy()
    for _ in range(future_steps):
        input_tensor = torch.FloatTensor(current_seq).unsqueeze(0)
        with torch.no_grad():
            pred = model(input_tensor).numpy()[0]
        predictions.append(pred)
        # 슬라이딩: 현재 시퀀스의 첫 행 제거, 새 예측값 추가
        current_seq = np.vstack([current_seq[1:], pred])
    predictions = np.array(predictions)
    predictions = scaler.inverse_transform(predictions)
    return predictions
