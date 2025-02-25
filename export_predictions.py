import torch
import pandas as pd
import numpy as np
from data_loader import load_data_from_mongo  # MongoDB에서 데이터를 불러옵니다.
from data_preprocessor import normalize_data, create_dataset, prepare_data  # 데이터 정규화 및 시계열 데이터셋 생성
from model import LSTMModel  # LSTM 모델 정의
from train import train_model  # 모델 학습 함수
from predict import predict_future  # 미래 예측 함수

def export_predictions():
    """
    [원리 설명]
    - MongoDB에서 데이터를 불러온 후, 날짜 전처리 및 숫자형 변환을 수행합니다.
    - 데이터를 정규화하고, 시계열 데이터셋을 생성하여 LSTM 모델을 학습합니다.
    - 학습된 모델을 사용해 미래 7일치 유가 예측을 수행하고, 
      역정규화를 통해 원래 스케일의 예측값을 복원한 후 CSV 파일로 저장합니다.
    
    순서:
      1) 데이터 로드 및 날짜 파싱
      2) 'date' 컬럼 제거 후, 모든 데이터를 숫자로 변환 및 결측치 처리
      3) 타겟 컬럼 존재 여부 확인
      4) 전체 데이터를 numpy 배열로 변환 후 정규화
      5) 시계열 데이터셋 생성
      6) LSTM 모델 구성 및 학습
      7) 미래 예측 수행 및 역정규화
      8) 예측 결과를 CSV 파일로 저장
    """
    # 1) MongoDB에서 데이터 불러오기
    df = load_data_from_mongo()

    # 2) 날짜 전처리: "Date_" 접두사 제거 및 "%Y_%m_%d" 포맷으로 파싱
    df['date'] = pd.to_datetime(df['date'].str.replace("Date_", "", regex=False), format="%Y_%m_%d")
    df.sort_values('date', inplace=True)
    
    # 3) 'date' 컬럼 제외 후 나머지 데이터를 사용
    features = df.drop(columns=['date'])
    features = features.apply(pd.to_numeric, errors='coerce')
    features = features.dropna(axis=1, how='all')
    
    # 4) 타겟 컬럼 설정 (유가 관련 4개 변수)
    target_cols = ["premiumGasoline", "gasoline", "diesel", "kerosene"]
    for col in target_cols:
        if col not in features.columns:
            raise ValueError(f"Target column '{col}' does not exist.")
    
    # 5) 전체 데이터를 배열로 변환 후 정규화
    data_array = features.values.astype(float)
    normalized_data, scaler = normalize_data(data_array)
    
    # 6) 시계열 데이터셋 생성 (look_back=3)
    look_back = 3
    X, Y_full = create_dataset(normalized_data, look_back)
    X, Y_full = prepare_data(X, Y_full)
    
    feature_columns = features.columns.tolist()
    target_indices = [feature_columns.index(col) for col in target_cols]
    
    Y_full_np = Y_full.numpy()
    Y = Y_full_np[:, target_indices]
    
    # 7) LSTM 모델 구성 및 학습
    input_dim = X.shape[2]
    hidden_dim = 50
    layer_dim = 1
    output_dim = len(target_cols)
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    model = train_model(model, X, torch.FloatTensor(Y))
    
    # 8) 미래 7일치 예측 수행
    future_steps = 7
    last_sequence = normalized_data[-look_back:]
    future_predictions_all = predict_future(model, last_sequence, future_steps, scaler)
    future_predictions = future_predictions_all[:, target_indices]
    
    # 9) 예측 결과를 CSV 파일로 저장
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)
    
    pred_df = pd.DataFrame(future_predictions, columns=target_cols, index=future_dates)
    pred_df.to_csv("fuel_predictions.csv")
    print("Predictions have been saved to fuel_predictions.csv")
    print(pred_df)

if __name__ == "__main__":
    export_predictions()
