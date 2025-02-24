import torch
import pandas as pd
import numpy as np
from data_loader import load_data_from_mongo
from data_preprocessor import normalize_data, create_dataset, prepare_data
from model import LSTMModel
from train import train_model
from predict import predict_future

def export_predictions():
    df = load_data_from_mongo()
    # 'date' 컬럼에서 "Date_" 접두사를 제거하고, "%Y_%m_%d" 형식으로 파싱
    df['date'] = pd.to_datetime(df['date'].str.replace("Date_", "", regex=False), format="%Y_%m_%d")
    df.sort_values('date', inplace=True)
    
    features = df.drop(columns=['date'])
    features = features.apply(pd.to_numeric, errors='coerce')
    features = features.dropna(axis=1, how='all')
    
    # 영어 컬럼명을 한글로 변환
    features = features.rename(columns={
        'premiumGasoline': '고급휘발유',
        'gasoline': '휘발유',
        'diesel': '경유',
        'kerosene': '등유'
    })
    
    target_cols = ["고급휘발유", "휘발유", "경유", "등유"]
    for col in target_cols:
        if col not in features.columns:
            raise ValueError(f"타겟 컬럼 '{col}'이 데이터에 존재하지 않습니다.")
    
    data_array = features.values.astype(float)
    normalized_data, scaler = normalize_data(data_array)
    
    look_back = 3
    X, Y_full = create_dataset(normalized_data, look_back)
    X, Y_full = prepare_data(X, Y_full)
    
    feature_columns = features.columns.tolist()
    target_indices = [feature_columns.index(col) for col in target_cols]
    
    Y_full_np = Y_full.numpy()
    Y = Y_full_np[:, target_indices]
    
    input_dim = X.shape[2]
    hidden_dim = 50
    layer_dim = 1
    output_dim = len(target_cols)
    
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    model = train_model(model, X, torch.FloatTensor(Y))
    
    future_steps = 7
    last_sequence = normalized_data[-look_back:]
    future_predictions_all = predict_future(model, last_sequence, future_steps, scaler)
    future_predictions = future_predictions_all[:, target_indices]
    
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)
    
    pred_df = pd.DataFrame(future_predictions, columns=target_cols, index=future_dates)
    pred_df.to_csv("korean_fuel_predictions.csv")
    print("예측 결과가 korean_fuel_predictions.csv 파일로 저장되었습니다.")
    print(pred_df)

if __name__ == "__main__":
    export_predictions()
