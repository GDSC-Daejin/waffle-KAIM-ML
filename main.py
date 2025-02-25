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
    리스트 형태(예: [1597.04, 1679.35, ...])의 데이터를 평균값으로 변환하는 함수입니다.
    리스트 안에 숫자로 바꿀 수 없는 값(문자열 등)이 있으면 무시하고,
    변환 가능한 숫자만 모아서 평균을 구합니다.
    만약 전혀 변환할 수 있는 값이 없다면 np.nan을 반환합니다.
    리스트가 아니면 float 변환을 시도하고, 실패하면 np.nan을 반환합니다.
    """
    if isinstance(val, list):
        numeric_vals = []
        for item in val:
            try:
                numeric_vals.append(float(item))
            except:
                # 숫자로 변환 불가능한 항목은 건너뜀
                pass
        if len(numeric_vals) == 0:
            return np.nan
        else:
            return float(np.mean(numeric_vals))
    else:
        try:
            return float(val)
        except:
            return np.nan

def create_dataset_for_XY(X_data, Y_data, look_back=3):
    """
    시계열 데이터셋을 생성하는 함수입니다.
    look_back만큼의 과거 데이터를 X로 하고, 그 다음 시점(다음 날 등)의 데이터를 Y로 둡니다.
    
    예:
      - look_back=3이면, 과거 3일 데이터를 이용해 4번째 날 데이터를 예측하도록 (X, Y) 구성.
    반환값: (X, Y) 형태의 numpy 배열
    """
    X, Y = [], []
    for i in range(len(X_data) - look_back):
        X.append(X_data[i : i + look_back])
        Y.append(Y_data[i + look_back])
    return np.array(X), np.array(Y)

def main():
    # 1) MongoDB에서 데이터 불러오기
    df = load_data_from_mongo()

    # 2) 날짜 처리: "Date_" 접두사를 제거하고 "%Y_%m_%d" 형태로 파싱
    df['date'] = pd.to_datetime(
        df['date'].str.replace("Date_", "", regex=False),
        format="%Y_%m_%d",
        errors='coerce'
    )
    # 날짜 오름차순 정렬
    df.sort_values('date', inplace=True)

    # (선택) 날짜를 1일 전으로 조정
    df['date'] = df['date'] - pd.Timedelta(days=1)

    # 3) 'date' 컬럼을 제거한 나머지 데이터를 features로 사용
    features = df.drop(columns=['date'])

    # 리스트를 평균값으로 변환
    for col in features.columns:
        features[col] = features[col].apply(parse_list_to_float)

    # 문자 등 숫자로 변환 불가능한 값 => NaN 처리, 전부 NaN인 컬럼은 삭제
    features = features.apply(pd.to_numeric, errors='coerce')
    features = features.dropna(axis=1, how='all')

    # 4) 예측 대상 컬럼(영어)
    target_cols = ["premiumGasoline", "gasoline", "diesel", "kerosene"]
    for col in target_cols:
        if col not in features.columns:
            raise ValueError(f"Target column '{col}' does not exist in the data.")

    # 나머지 컬럼은 입력으로 활용
    all_cols = list(features.columns)
    input_cols = [c for c in all_cols if c not in target_cols]

    # 5) 입력(X), 타겟(Y) 데이터를 numpy 배열로 변환 후 정규화
    X_data = features[input_cols].values.astype(float)
    Y_data = features[target_cols].values.astype(float)

    scaler_input = MinMaxScaler()
    scaler_target = MinMaxScaler()

    X_data_scaled = scaler_input.fit_transform(X_data)
    Y_data_scaled = scaler_target.fit_transform(Y_data)

    # 6) look_back 길이를 활용해 시계열 (X, Y) 데이터셋 생성
    look_back = 3
    X_np, Y_np = create_dataset_for_XY(X_data_scaled, Y_data_scaled, look_back=look_back)
    X_torch = torch.FloatTensor(X_np)
    Y_torch = torch.FloatTensor(Y_np)

    # 7) LSTM 모델 구성 및 학습
    input_dim = X_torch.shape[2]
    hidden_dim = 50
    layer_dim = 1
    output_dim = Y_torch.shape[1]  # 타겟 컬럼이 4개

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    model = train_model(model, X_torch, Y_torch, epochs=100, batch_size=32)

    # 8) 학습 데이터 예측 및 역정규화
    preds_scaled = make_predictions(model, X_torch)
    preds = scaler_target.inverse_transform(preds_scaled)
    Y_actual_scaled = Y_torch.numpy()
    Y_actual = scaler_target.inverse_transform(Y_actual_scaled)

    # 예측 정확도 측정(RMSE)
    rmse = calculate_rmse(Y_actual, preds)
    print(f"RMSE: {rmse:.2f}")

    # 9) 미래 예측 (7일치)
    future_steps = 7
    last_seq = X_data_scaled[-look_back:]

    model.eval()
    future_preds_scaled = []
    with torch.no_grad():
        current_seq = last_seq.copy()
        for _ in range(future_steps):
            inp = torch.FloatTensor(current_seq).unsqueeze(0)
            pred_s = model(inp).numpy()[0]
            future_preds_scaled.append(pred_s)
            # 슬라이딩 로직 (현재 구현은 입력 크기 vs. 출력 크기가 달라 오류 발생 가능)
            current_seq = np.vstack([current_seq[1:], current_seq[-1]])

    future_preds_scaled = np.array(future_preds_scaled)
    future_preds = scaler_target.inverse_transform(future_preds_scaled)

    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)

    # 10) 시각화
    used_dates = df['date'].iloc[look_back:]
    hist_data = Y_data[-(len(used_dates)):]
    hist_data_inv = scaler_target.inverse_transform(hist_data)

    plot_results(used_dates, hist_data_inv, future_dates, future_preds)

if __name__ == "__main__":
    main()
