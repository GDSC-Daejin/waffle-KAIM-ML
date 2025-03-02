#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
멀티 모델 앙상블 유가 예측 모델 (수정 버전)
- 상대 경로 사용, 캐싱 적용
- ARIMA, Prophet, XGBoost, LightGBM, LSTM 등 다양한 모델을 사용하여
  검증 기반 앙상블 예측을 통해 예측 정확도를 향상시킵니다.
- 최종적으로 전국 및 지역별(예: gasoline_National, gasoline_Seoul 등) 일주일치 예측 결과 반환
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import joblib
import time
import json

# 모델링 라이브러리
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

import xgboost as xgb
import lightgbm as lgb

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Prophet 라이브러리 (설치되어 있으면 사용)
try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    print("Prophet 라이브러리가 설치되어 있지 않습니다.")
    prophet_available = False

warnings.filterwarnings('ignore')

# 상대 경로 설정
OUTPUT_DIR = './results'
CACHE_DIR = './cache'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# joblib 캐시 설정 (RAM 여유 활용)
memory = joblib.Memory(location=CACHE_DIR, verbose=0)

# 시드 설정
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {DEVICE}")

# =============================================================================
# 1. 데이터 로드 및 전처리 (캐싱 적용)
# =============================================================================
@memory.cache
def load_and_preprocess_data(file_path):
    print(f"데이터 로드 중: {file_path}")
    df = pd.read_csv(file_path)
    print(f"원본 데이터 크기: {df.shape}")
    
    # 날짜 처리: "Date_" 접두어 제거
    df['date'] = pd.to_datetime(df['date'].str.replace('Date_', ''), format='%Y_%m_%d')
    df = df.sort_values('date').reset_index(drop=True)
    
    # 지역별 연료 가격 데이터 처리
    regions = ['National', 'Seoul', 'Busan', 'Daegu', 'Incheon', 'Gwangju', 'Daejeon', 'Ulsan', 
               'Sejong', 'Gyeonggi', 'Gangwon', 'Chungbuk', 'Chungnam', 'Jeonbuk', 'Jeonnam', 
               'Gyeongbuk', 'Gyeongnam']
    for fuel_type in ['gasoline', 'premiumGasoline', 'diesel', 'kerosene']:
        if fuel_type in df.columns and isinstance(df[fuel_type].iloc[0], str):
            df[fuel_type] = df[fuel_type].apply(lambda x: eval(x) if isinstance(x, str) else x)
            if isinstance(df[fuel_type].iloc[0], list) and len(df[fuel_type].iloc[0]) == len(regions):
                for i, region in enumerate(regions):
                    df[f'{fuel_type}_{region}'] = df[fuel_type].apply(lambda x: float(x[i]) if isinstance(x, list) else np.nan)
            df = df.drop(fuel_type, axis=1)
    
    if 'area' in df.columns and isinstance(df['area'].iloc[0], str):
        df = df.drop('area', axis=1)
    
    # 자료형 최적화
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
            
    # 수치형 컬럼
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # 예측 대상으로 사용할 전국 및 지역별 가격 컬럼 (예: gasoline_National, gasoline_Seoul 등)
    gasoline_columns = [col for col in df.columns if col.startswith('gasoline_')]
    
    print("예측 대상 변수:", gasoline_columns)
    print("\n데이터 기간:", df['date'].min().strftime('%Y-%m-%d'), "부터", 
          df['date'].max().strftime('%Y-%m-%d'))
    print(f"총 {df.shape[0]}일 데이터")
    return df, numeric_cols, gasoline_columns

# =============================================================================
# 2. 특성 엔지니어링 (캐싱 적용)
# =============================================================================
@memory.cache
def create_features(df, target_col):
    print("\n=== 특성 엔지니어링 시작 ===")
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 예시: 간단한 이동평균, EMA, lag 변수 생성 (추가 특성은 필요에 따라 확대)
    for feature in [target_col]:
        for window in [3, 7, 14]:
            df[f'{feature}_MA{window}'] = df[target_col].rolling(window=window).mean()
        for lag in [1, 2, 3]:
            df[f'{feature}_lag{lag}'] = df[target_col].shift(lag)
    
    df = df.dropna()
    print(f"특성 생성 후 데이터 크기: {df.shape}")
    print("=== 특성 엔지니어링 완료 ===")
    return df

# =============================================================================
# 3. 딥러닝 모델 (LSTM) 정의 및 학습
# =============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim//2, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        out = self.fc1(last_time_step)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def train_lstm_model(X_train, y_train, X_val, y_val, input_dim, epochs=50, batch_size=32, lr=0.001):
    model = LSTMModel(input_dim=input_dim, hidden_dim=64, num_layers=2).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_val_loss = np.inf
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val.to(DEVICE)).squeeze(), y_val.to(DEVICE)).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
        print(f"LSTM Epoch {epoch+1}: Train Loss={np.mean(train_losses):.4f}, Val Loss={val_loss:.4f}")
    model.load_state_dict(best_model)
    return model

# =============================================================================
# 4. 각 모델별 예측 함수
# =============================================================================
def forecast_arima(series, forecast_horizon=7, order=(5,1,0), seasonal_order=(0,0,0,0)):
    """ ARIMA를 이용한 예측 """
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    pred = fitted.get_forecast(steps=forecast_horizon)
    return pred.predicted_mean

def forecast_prophet(df_prophet, forecast_horizon=7):
    """ Prophet을 이용한 예측 """
    if not prophet_available:
        print("Prophet 라이브러리 미설치로 예측 불가")
        return None
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    # 주요 한국 휴일 추가 가능
    m.add_country_holidays(country_name='South Korea')
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=forecast_horizon, freq='D')
    forecast = m.predict(future)
    # Prophet 예측 중 forecast 기간에 해당하는 yhat 반환
    yhat = forecast.tail(forecast_horizon)['yhat']
    return yhat

def forecast_ml_model(model, last_features, forecast_steps=7, seq_length=30):
    """
    XGBoost, LightGBM 등의 머신러닝 모델에 대해
    마지막 seq_length일의 데이터를 바탕으로 재귀적으로 forecast.
    last_features: 마지막 시퀀스(평탄화된 특성 벡터)
    이 함수는 단순 예시이며, 실제 구현 시에는 미래 날짜의 캘린더 및 파생변수 생성 로직이 필요합니다.
    """
    preds = []
    current_seq = last_features.copy()
    for _ in range(forecast_steps):
        # 예측: 현재 평탄화된 시퀀스 사용
        pred = model.predict(current_seq.reshape(1, -1))[0]
        preds.append(pred)
        # 재귀적으로 다음 입력 갱신 (여기서는 가장 간단하게 마지막 값 교체)
        current_seq = np.roll(current_seq, -1)
        current_seq[-1] = pred
    return np.array(preds)

# =============================================================================
# 5. 검증을 통한 가중치 산출 함수
# =============================================================================
def optimize_model_weights(pred_dict, y_true):
    """ 각 모델의 MAPE 역수를 가중치로 산출 """
    weights = {}
    for name, pred in pred_dict.items():
        mape = mean_absolute_percentage_error(y_true, pred) * 100
        weights[name] = 1/(mape + 1e-10)
    total = sum(weights.values())
    normalized = {k: v/total for k, v in weights.items()}
    return normalized

# =============================================================================
# 6. 멀티 모델 예측 및 앙상블 함수
# =============================================================================
def multi_model_forecasting(series, df_prophet, ml_train_X, ml_train_y, ml_model_xgb, ml_model_lgb, lstm_model,
                            forecast_horizon=7, seq_length=30):
    """
    시계열 series에 대해 여러 모델로 예측하고, 검증 후 앙상블
    인자:
      series: pd.Series (날짜 인덱스)
      df_prophet: Prophet용 데이터프레임 (ds, y)
      ml_train_X, ml_train_y: 머신러닝 모델 학습용 데이터 (예: X_ml, y_ml)
      ml_model_xgb, ml_model_lgb: 사전 학습된 머신러닝 모델
      lstm_model: 사전 학습된 딥러닝 모델 (입력 형태에 맞게 준비)
    """
    # 1) ARIMA 예측
    pred_arima = forecast_arima(series, forecast_horizon)
    
    # 2) Prophet 예측 (if available)
    if prophet_available:
        pred_prophet = forecast_prophet(df_prophet, forecast_horizon)
    else:
        pred_prophet = None
    
    # 3) 머신러닝 예측 (XGBoost, LightGBM)
    # 여기서는 ml_train_X의 마지막 샘플(최근 seq_length 일)을 사용하여 재귀적 forecast 수행
    last_features = ml_train_X[-1]  # 평탄화된 특성 벡터
    pred_xgb = forecast_ml_model(ml_model_xgb, last_features, forecast_steps=forecast_horizon, seq_length=seq_length)
    pred_lgb = forecast_ml_model(ml_model_lgb, last_features, forecast_steps=forecast_horizon, seq_length=seq_length)
    
    # 4) 딥러닝 예측 (LSTM)
    # 마지막 seq_length 데이터를 가져와서 텐서 변환 후 예측 (여기서는 간단 예시)
    # (실제 입력 형태에 맞게 reshape 필요)
    last_seq = ml_train_X[-1].reshape(1, seq_length, -1)  # 예시: (1, seq_length, feature_dim)
    lstm_model.eval()
    with torch.no_grad():
        last_seq_tensor = torch.FloatTensor(last_seq).to(DEVICE)
        pred_lstm = lstm_model(last_seq_tensor).cpu().numpy().flatten()
        # 단일 스텝 예측을 반복하여 forecast_horizon 만큼 확장 가능 (여기서는 단순 반복 사용)
        pred_lstm = np.repeat(pred_lstm, forecast_horizon)
    
    # 5) 검증 단계: 만약 과거 검증 데이터를 이용해 각 모델의 성능을 평가할 수 있다면, 
    #    그 결과로 가중치를 산출합니다. 여기서는 예시로 모든 모델에 동일 가중치를 부여합니다.
    model_preds = {
        'ARIMA': pred_arima.values,
        'XGBoost': pred_xgb,
        'LightGBM': pred_lgb,
        'LSTM': pred_lstm
    }
    if pred_prophet is not None:
        model_preds['Prophet'] = pred_prophet.values
    
    # (검증용 실제 y_true가 있다면 optimize_model_weights() 호출)
    # 여기서는 동일 가중치 사용
    num_models = len(model_preds)
    weights = {name: 1/num_models for name in model_preds.keys()}
    print("사용된 모델 가중치:", weights)
    
    # 6) 앙상블 예측
    ensemble_pred = np.zeros(forecast_horizon)
    for name, pred in model_preds.items():
        ensemble_pred += weights[name] * pred
    
    # 중간 모델별 예측 결과 print (검증용)
    for name, pred in model_preds.items():
        print(f"{name} 예측: {pred}")
    
    return ensemble_pred, model_preds

# =============================================================================
# 7. 최종 예측: 일주일치 전국 및 지역별 예측 수행
# =============================================================================
def forecast_one_week(df, numeric_cols, gasoline_columns, seq_length=30):
    print("\n=== 일주일치 유가 예측 (멀티 모델 앙상블) 시작 ===")
    final_forecasts = {}
    last_date = df['date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
    print("예측 날짜:", future_dates)
    
    # for each target column (전국, 지역별)
    for target_col in gasoline_columns:
        print(f"\n[{target_col}] 모델 학습 및 예측 진행 중...")
        ts = df.set_index('date')[target_col].dropna()
        if len(ts) < 50:
            print(f"데이터 부족으로 {target_col} 예측 건너뜀")
            continue
        # 분할: 마지막 7일을 검증용으로 사용 (여기서는 단순 분할)
        train_series = ts[:-7]
        val_series = ts[-7:]
        
        # Prophet용 데이터 준비
        df_prophet = pd.DataFrame({'ds': train_series.index, 'y': train_series.values})
        
        # 머신러닝 모델 학습용 데이터 준비
        # (prepare_time_series_data와 유사하게, 여기서는 평탄화된 특성 벡터 생성)
        # 실제로는 seq_length, lag, 이동평균 등을 포함한 특성 생성 함수 사용 권장
        X_ml = []
        y_ml = []
        values = train_series.values
        for i in range(seq_length, len(values)):
            X_ml.append(values[i-seq_length:i].flatten())
            y_ml.append(values[i])
        X_ml = np.array(X_ml)
        y_ml = np.array(y_ml)
        
        # Train/Validation split: 마지막 7일 검증
        train_size = len(X_ml) - 7
        X_train_ml, y_train_ml = X_ml[:train_size], y_ml[:train_size]
        X_val_ml, y_val_ml = X_ml[train_size:], y_ml[train_size:]
        
        # 머신러닝 모델 학습: XGBoost & LightGBM
        model_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.01, random_state=RANDOM_SEED)
        model_xgb.fit(X_train_ml, y_train_ml)
        model_lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.01, random_state=RANDOM_SEED)
        model_lgb.fit(X_train_ml, y_train_ml)
        
        # 딥러닝 모델 학습 (LSTM)
        # reshape X_train_ml to (samples, seq_length, 1) for LSTM
        X_train_dl = X_train_ml.reshape(-1, seq_length, 1)
        X_val_dl = X_val_ml.reshape(-1, seq_length, 1)
        y_train_dl = torch.FloatTensor(y_train_ml)
        y_val_dl = torch.FloatTensor(y_val_ml)
        X_train_dl_tensor = torch.FloatTensor(X_train_dl)
        X_val_dl_tensor = torch.FloatTensor(X_val_dl)
        lstm_model = train_lstm_model(X_train_dl_tensor, y_train_dl, X_val_dl_tensor, y_val_dl, input_dim=1, epochs=30)
        
        # 검증 단계: 각 모델 예측 (선택 사항: 실제 검증 지표 산출)
        # 여기서는 검증 단계 생략하고, 전체 train_series를 이용하여 forecast 진행
        ensemble_pred, model_preds = multi_model_forecasting(train_series, df_prophet, X_ml, y_ml,
                                                             model_xgb, model_lgb, lstm_model,
                                                             forecast_horizon=7, seq_length=seq_length)
        print(f"{target_col} 최종 앙상블 예측:")
        print(ensemble_pred)
        final_forecasts[target_col] = pd.Series(ensemble_pred, index=future_dates)
    
    print("\n=== 일주일치 유가 예측 완료 ===")
    return final_forecasts

# =============================================================================
# 메인 실행부
# =============================================================================
def main():
    file_path = './korea_economic_data.csv'
    df, numeric_cols, gasoline_columns = load_and_preprocess_data(file_path)
    # 중간 점검: 데이터 일부 출력
    print("\n데이터 일부:")
    print(df.head())
    
    # 특성 엔지니어링 (대표 변수: gasoline_National)
    df_features = create_features(df, 'gasoline_National')
    print("\n특성 엔지니어링 완료 후 데이터 일부:")
    print(df_features.head())
    
    # 변수 중요도 분석 (전국 가격 기준)
    # analyze_feature_importance() 기존 함수 활용 가능 (생략)
    
    # 최종 예측: 일주일치 (전국 및 지역별)
    forecasts = forecast_one_week(df_features, numeric_cols, gasoline_columns, seq_length=30)
    print("\n=== 최종 예측 결과 (일주일치) ===")
    for col, pred in forecasts.items():
        print(f"\n{col} 예측:")
        print(pred)
    
    # 결과 시각화 (필요 시 추가 구현)
  
if __name__ == '__main__':
    main()
