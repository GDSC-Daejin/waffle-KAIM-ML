import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from data_loader import load_data_from_mongo  # MongoDB에서 데이터 불러오는 함수
from data_preprocessor import prepare_data
from model import LSTMModel
from train import train_model
from predict import make_predictions, calculate_rmse
from visualize import plot_results

def create_dataset_for_XY(X_data, Y_data, look_back=3):
    """
    입력(X_data)과 타겟(Y_data)을 이용해 시계열 (X, Y) 쌍을 생성합니다.
    X: (batch_size, look_back, num_input_features)
    Y: (batch_size, num_target_features)
    """
    X, Y = [], []
    for i in range(len(X_data) - look_back):
        X.append(X_data[i:i+look_back])
        Y.append(Y_data[i+look_back])
    return np.array(X), np.array(Y)

def main():
    # 1) MongoDB에서 데이터 불러오기
    df = load_data_from_mongo()
    print("=== MongoDB에서 불러온 원본 데이터 ===")
    print(df.head(10))  # 부분만 출력 (소수점 표시를 늘리고 싶으면 pd.set_option 사용)

    # 2) 날짜 처리 (date 컬럼 파싱)
    # - DB에 'date'가 "Date_YYYY_MM_DD" 형식으로 들어있다고 가정
    df['date'] = pd.to_datetime(
        df['date'].str.replace("Date_", "", regex=False),
        format="%Y_%m_%d",
        errors='coerce'
    )
    df.sort_values('date', inplace=True)

    # 필요 시 날짜를 하루 전으로 조정
    df['date'] = df['date'] - pd.Timedelta(days=1)

    # 3) date 컬럼 제외 -> 모든 나머지 컬럼을 일단 입력 후보
    #   (MongoDB 데이터 중, date만 제외)
    features = df.drop(columns=['date'])

    # 숫자로 변환 시도: 문자열이거나 변환 불가면 NaN
    features = features.apply(pd.to_numeric, errors='coerce')

    # 전부 NaN인 열은 제거하지 않으려면 아래 라인을 주석 처리
    # (만약 컬럼 전체가 NaN이면 학습에 사용할 수 없긴 합니다)
    # features = features.dropna(axis=1, how='all')

    print("=== 전처리 후 컬럼 목록 ===")
    print(features.columns.tolist())

    # 4) 우리가 예측할 4가지 타겟 컬럼
    target_cols = ["premiumGasoline", "gasoline", "diesel", "kerosene"]
    for col in target_cols:
        if col not in features.columns:
            raise ValueError(f"타겟 컬럼 '{col}'이 데이터에 존재하지 않습니다.")

    # 5) 입력(Features) vs 타겟(Target) 분리
    all_cols = list(features.columns)
    input_cols = [c for c in all_cols if c not in target_cols]
    print("입력에 사용될 컬럼들:", input_cols)
    print("타겟 컬럼들:", target_cols)

    # 6) NumPy 변환
    X_data = features[input_cols].values.astype(float)      # shape: (N, num_input_features)
    Y_data = features[target_cols].values.astype(float)     # shape: (N, 4)

    # 7) 입력/타겟 각각 MinMaxScaler
    scaler_input = MinMaxScaler()
    scaler_target = MinMaxScaler()

    X_data_scaled = scaler_input.fit_transform(X_data)  # (N, num_input_features)
    Y_data_scaled = scaler_target.fit_transform(Y_data) # (N, 4)

    # 8) 시계열 Dataset 생성
    look_back = 3
    X, Y = create_dataset_for_XY(X_data_scaled, Y_data_scaled, look_back=look_back)

    # torch 텐서로 변환
    X, Y = torch.FloatTensor(X), torch.FloatTensor(Y)

    # 9) 모델 정의 및 학습
    input_dim = X.shape[2]  # num_input_features
    hidden_dim = 50
    layer_dim = 1
    output_dim = Y.shape[1]  # 4
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

    model = train_model(model, X, Y, epochs=100, batch_size=32)

    # 10) 학습 데이터 예측 & 역정규화
    preds_scaled = make_predictions(model, X)            # shape: (N - look_back, 4)
    preds = scaler_target.inverse_transform(preds_scaled)
    Y_actual_scaled = Y.numpy()                          # shape: (N - look_back, 4)
    Y_actual = scaler_target.inverse_transform(Y_actual_scaled)

    rmse = calculate_rmse(Y_actual, preds)
    print(f"RMSE: {rmse:.2f}")

    # 11) 미래 예측
    future_steps = 7
    # 마지막 look_back 구간의 입력 시퀀스
    last_seq_X = X_data_scaled[-look_back:]  # shape: (look_back, num_input_features)

    # 직접 future 예측 로직
    predictions_future_scaled = []
    current_seq = last_seq_X.copy()
    model.eval()
    with torch.no_grad():
        for _ in range(future_steps):
            inp = torch.FloatTensor(current_seq).unsqueeze(0)  # (1, look_back, input_dim)
            pred_s = model(inp).numpy()[0]  # (4,) scaled
            predictions_future_scaled.append(pred_s)
            # 여기서는 입력 시퀀스가 변하지 않는다고 가정
            # (만약 타겟 예측값을 다시 X에 반영해야 한다면 별도 로직 필요)
            current_seq = np.vstack([current_seq[1:], current_seq[-1]])

    predictions_future_scaled = np.array(predictions_future_scaled)  # (future_steps, 4)
    predictions_future = scaler_target.inverse_transform(predictions_future_scaled)

    # 날짜 생성: 마지막 date 이후로 7일
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)

    # 12) 시각화
    #   - 실제값 중 (N - look_back) 구간이 preds와 1:1 매칭
    #   - 필요 시 전체 과거를 그리고 싶다면 look_back 보정
    #   - 여기서는 편의상 look_back 이후의 날짜만 일치시킴
    from visualize import plot_results
    historical_actual = Y_data[-(len(Y_actual)):]  # shape: (N-look_back, 4)
    # 역정규화
    historical_actual = scaler_target.inverse_transform(historical_actual)

    used_dates = df['date'].iloc[look_back:]  # (N - look_back)
    plot_results(used_dates, historical_actual, future_dates, predictions_future)

if __name__ == "__main__":
    main()
n()
