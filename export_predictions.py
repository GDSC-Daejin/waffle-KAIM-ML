import torch
import pandas as pd
import numpy as np
from data_loader import load_data_from_mongo
from data_preprocessor import normalize_data, create_dataset, prepare_data
from model import LSTMModel
from train import train_model
from predict import predict_future

def export_predictions():
    """
    MongoDB에서 데이터를 불러와 미래 예측 결과를 CSV로 내보냅니다.
    """
    # 1) MongoDB에서 데이터 불러오기
    df = load_data_from_mongo()

    # 2) 날짜 처리 ("Date_" 접두사 제거 + "%Y_%m_%d" 형태)
    df['date'] = pd.to_datetime(df['date'].str.replace("Date_", "", regex=False), format="%Y_%m_%d")
    df.sort_values('date', inplace=True)
    
    # 3) date 컬럼 제외한 나머지
    features = df.drop(columns=['date'])
    # 숫자로 변환 (변환 안 되면 NaN), 전부 NaN이면 컬럼 제거
    features = features.apply(pd.to_numeric, errors='coerce')
    features = features.dropna(axis=1, how='all')
    
    # 타겟 컬럼 (영어)
    target_cols = ["premiumGasoline", "gasoline", "diesel", "kerosene"]
    for col in target_cols:
        if col not in features.columns:
            raise ValueError(f"Target column '{col}' does not exist.")

    # 4) 배열 변환 후 정규화
    data_array = features.values.astype(float)
    normalized_data, scaler = normalize_data(data_array)
    
    # 5) 시계열 데이터셋 생성
    look_back = 3
    X, Y_full = create_dataset(normalized_data, look_back)
    X, Y_full = prepare_data(X, Y_full)
    
    feature_columns = features.columns.tolist()
    target_indices = [feature_columns.index(col) for col in target_cols]
    
    Y_full_np = Y_full.numpy()
    Y = Y_full_np[:, target_indices]
    
    # 6) 모델 생성 및 학습
    input_dim = X.shape[2]
    hidden_dim = 50
    layer_dim = 1
    output_dim = len(target_cols)
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    model = train_model(model, X, torch.FloatTensor(Y))
    
    # 7) 미래 예측 (7일)
    future_steps = 7
    last_sequence = normalized_data[-look_back:]
    future_predictions_all = predict_future(model, last_sequence, future_steps, scaler)
    future_predictions = future_predictions_all[:, target_indices]
    
    # 8) 예측 결과를 CSV로 내보내기
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)
    
    pred_df = pd.DataFrame(future_predictions, columns=target_cols, index=future_dates)
    pred_df.to_csv("fuel_predictions.csv")
    #pred_df.to_csv("/path/to/your/folder/fuel_predictions.csv")
    print("Predictions have been saved to fuel_predictions.csv")
    print(pred_df)

if __name__ == "__main__":
    export_predictions()
