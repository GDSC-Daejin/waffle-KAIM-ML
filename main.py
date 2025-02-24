import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from data_loader import load_data_from_mongo
from model import LSTMModel
from train import train_model
from predict import make_predictions, calculate_rmse
from visualize import plot_results

def parse_list_to_float(val):
    """
    리스트 형태인 컬럼(예: [1597.04, 1679.35, ...])을 평균값으로 변환.
    리스트 내부에 숫자 아닌 문자열이 섞여 있다면 변환 가능한 숫자만 취합해 평균 계산.
    숫자가 전혀 없으면 np.nan.
    리스트가 아니면 float 변환 시도, 실패 시 np.nan.
    """
    if isinstance(val, list):
        numeric_vals = []
        for item in val:
            try:
                numeric_vals.append(float(item))
            except:
                pass
        if len(numeric_vals) == 0:
            return np.nan
        else:
            return float(np.mean(numeric_vals))
    else:
        # 리스트가 아닐 경우 -> 문자열이면 float 변환 시도
        try:
            return float(val)
        except:
            return np.nan

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
    
    # print("=== MongoDB에서 불러온 원본 데이터 (일부) ===")
    # print(df.head(10))

    # 2) 날짜 처리
    df['date'] = pd.to_datetime(
        df['date'].str.replace("Date_", "", regex=False),
        format="%Y_%m_%d",
        errors='coerce'
    )
    df.sort_values('date', inplace=True)
    
    # 날짜를 하루 전으로 조정 (필요 시)
    df['date'] = df['date'] - pd.Timedelta(days=1)

    # 3) 'date' 제외 -> features
    features = df.drop(columns=['date'])

    # (디버깅) 전처리 전 컬럼명
    # print("=== 전처리 전 컬럼 목록 ===")
    # print(features.columns.tolist())

    # 4) 리스트 -> float 변환
    for col in features.columns:
        features[col] = features[col].apply(parse_list_to_float)

    # 5) 숫자 변환 (문자열 → float, 실패 시 NaN)
    features = features.apply(pd.to_numeric, errors='coerce')

    # 전부 NaN인 열 제거(원한다면 주석 처리 가능)
    features = features.dropna(axis=1, how='all')

    # (디버깅) 전처리 후 컬럼명
    # print("=== 전처리 후 컬럼 목록 ===")
    # print(features.columns.tolist())

    # print("=== 전처리 후 데이터 (head) ===")
    # print(features.head(10))

    # 6) 네 가지 타겟 컬럼
    target_cols = ["premiumGasoline", "gasoline", "diesel", "kerosene"]
    for col in target_cols:
        if col not in features.columns:
            raise ValueError(f"타겟 컬럼 '{col}'이 데이터에 존재하지 않습니다.")

    # 7) 나머지 = 입력 컬럼
    all_cols = list(features.columns)
    input_cols = [c for c in all_cols if c not in target_cols]

    print("입력 컬럼:", input_cols)
    print("타겟 컬럼:", target_cols)

    # 8) 입력 / 타겟 데이터 분리
    X_data = features[input_cols].values.astype(float)   # (N, num_input_features)
    Y_data = features[target_cols].values.astype(float)  # (N, 4)

    # 9) MinMaxScaler 분리(입력용 / 타겟용)
    scaler_input = MinMaxScaler()
    scaler_target = MinMaxScaler()
    
    X_data_scaled = scaler_input.fit_transform(X_data)
    Y_data_scaled = scaler_target.fit_transform(Y_data)

    # 10) 시계열 Dataset 생성
    look_back = 3
    X_np, Y_np = create_dataset_for_XY(X_data_scaled, Y_data_scaled, look_back=look_back)
    X_torch = torch.FloatTensor(X_np)
    Y_torch = torch.FloatTensor(Y_np)

    # 11) 모델 생성 및 학습
    input_dim = X_torch.shape[2]
    hidden_dim = 50
    layer_dim = 1
    output_dim = Y_torch.shape[1]  # 4

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    model = train_model(model, X_torch, Y_torch, epochs=100, batch_size=32)

    # 12) 학습 데이터 예측 -> 역정규화
    preds_scaled = make_predictions(model, X_torch)  # shape: (N-look_back, 4)
    preds = scaler_target.inverse_transform(preds_scaled)
    Y_actual_scaled = Y_torch.numpy()
    Y_actual = scaler_target.inverse_transform(Y_actual_scaled)

    rmse = calculate_rmse(Y_actual, preds)
    print(f"RMSE: {rmse:.2f}")

    # 13) 미래 예측 (7일)
    future_steps = 7
    # 마지막 look_back 구간의 입력 시퀀스 (shape: (look_back, num_input_features))
    last_seq = X_data_scaled[-look_back:]

    model.eval()
    future_preds_scaled = []
    with torch.no_grad():
        current_seq = last_seq.copy()
        for _ in range(future_steps):
            inp = torch.FloatTensor(current_seq).unsqueeze(0)  # (1, look_back, input_dim)
            pred_s = model(inp).numpy()[0]  # (4,)
            future_preds_scaled.append(pred_s)
            # 입력 시퀀스(특히 타겟 값 등)는 갱신하지 않는다고 가정(단순 시나리오)
            current_seq = np.vstack([current_seq[1:], current_seq[-1]])

    future_preds_scaled = np.array(future_preds_scaled)
    future_preds = scaler_target.inverse_transform(future_preds_scaled)

    # 날짜 생성
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)

    # 14) 시각화
    # 모델 학습 시 look_back만큼 감소 -> (N-look_back)개가 예측과 매칭
    used_dates = df['date'].iloc[look_back:]  # shape: (N-look_back,)
    # Y_data[-(len(used_dates)):] => 마지막 (N-look_back)개 행
    hist_data = Y_data[-(len(used_dates)):]
    hist_data_inv = scaler_target.inverse_transform(hist_data)

    plot_results(used_dates, hist_data_inv, future_dates, future_preds)

if __name__ == "__main__":
    main()
