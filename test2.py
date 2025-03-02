#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
멀티 모델 앙상블 유가 예측 모델 (최종 수정 버전)
- 상대 경로 사용, 캐싱 적용
- ARIMA, Prophet, XGBoost, LightGBM, LSTM 등 다양한 모델을 사용하여
  검증 기반 앙상블 예측으로 예측 정확도를 향상시킵니다.
- 전국 및 지역별(예: gasoline_National, gasoline_Seoul 등) 그리고
  유종별(휘발유, 경유, 고급휘발유, 등유) 일주일치 예측 결과를 반환합니다.
- 비동기(병렬) 처리를 통해 여러 작업을 동시에 실행하며, 로그는 파일에 기록됩니다.
"""

import os, logging, time, json, warnings, joblib, numpy as np, pandas as pd
from datetime import datetime, timedelta
import concurrent.futures

# 시각화 (필요 시)
import matplotlib.pyplot as plt
import seaborn as sns

# 모델링 라이브러리
import torch, torch.nn as nn, torch.optim as optim
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

# 상대 경로 설정 및 디렉토리 생성
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

# 로깅 설정
logging.basicConfig(
    filename='./forecast.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# 1. 데이터 로드 및 전처리 (캐싱 적용)
# =============================================================================
@memory.cache
def load_and_preprocess_data(file_path):
    logger.info(f"데이터 로드 중: {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"원본 데이터 크기: {df.shape}")
    
    # 날짜 처리: "Date_" 접두어 제거 후 정렬
    df['date'] = pd.to_datetime(df['date'].str.replace('Date_', ''), format='%Y_%m_%d')
    df = df.sort_values('date').reset_index(drop=True)
    
    # 지역별 연료 가격 데이터 처리 (문자열 형태의 리스트를 실제 값으로 변환)
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
    
    # 수치형 컬럼 및 유종별 열 추출
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    fuel_types = ['gasoline', 'premiumGasoline', 'diesel', 'kerosene']
    fuel_columns = {}
    for ft in fuel_types:
        fuel_columns[ft] = [col for col in df.columns if col.startswith(f"{ft}_")]
    
    logger.info(f"예측 대상 변수: {fuel_columns}")
    logger.info(f"데이터 기간: {df['date'].min().strftime('%Y-%m-%d')} 부터 {df['date'].max().strftime('%Y-%m-%d')}, 총 {df.shape[0]}일")
    return df, numeric_cols, fuel_columns

# =============================================================================
# 2. 특성 엔지니어링 (캐싱 적용)
# =============================================================================
@memory.cache
def create_features(df, target_col):
    logger.info("=== 특성 엔지니어링 시작 ===")
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    
    # 간단한 이동평균 및 lag 특성 생성 (필요에 따라 확장)
    for feature in [target_col]:
        for window in [3, 7, 14]:
            df[f'{feature}_MA{window}'] = df[target_col].rolling(window=window).mean()
        for lag in [1,2,3]:
            df[f'{feature}_lag{lag}'] = df[target_col].shift(lag)
    
    df = df.dropna()
    logger.info(f"특성 생성 후 데이터 크기: {df.shape}")
    logger.info("=== 특성 엔지니어링 완료 ===")
    return df

# =============================================================================
# 3. 딥러닝 모델 (LSTM) 정의 및 학습
# =============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
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

def train_lstm_model(X_train, y_train, X_val, y_val, input_dim, epochs=30, batch_size=32, lr=0.001):
    model = LSTMModel(input_dim=input_dim, hidden_dim=64, num_layers=2).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    best_val_loss = np.inf
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val.to(DEVICE)).squeeze(), y_val.to(DEVICE)).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
        print(f"LSTM Epoch {epoch+1}: Train Loss={np.mean(losses):.4f}, Val Loss={val_loss:.4f}")
    model.load_state_dict(best_state)
    return model

# =============================================================================
# 4. 각 모델별 예측 함수
# =============================================================================
def forecast_arima(series, forecast_horizon=7, order=(5,1,0), seasonal_order=(0,0,0,0)):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    pred = fitted.get_forecast(steps=forecast_horizon)
    return pred.predicted_mean

# Prophet – 공휴일 처리 개선
def get_korean_holidays(years):
    holidays = []
    for year in years:
        holidays.append({'holiday': 'New Year', 'ds': f'{year}-01-01', 'lower_window': 0, 'upper_window': 1})
    return pd.DataFrame(holidays)

def forecast_prophet(df_prophet, forecast_horizon=7):
    if not prophet_available:
        print("Prophet 라이브러리 미설치로 예측 불가")
        return None
    years = list(range(df_prophet['ds'].dt.year.min(), df_prophet['ds'].dt.year.max()+2))
    holidays_df = get_korean_holidays(years)
    try:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                    holidays=holidays_df)
    except Exception as e:
        print(f"Prophet 오류: {e}")
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=forecast_horizon, freq='D')
    forecast = m.predict(future)
    return forecast.tail(forecast_horizon)['yhat']

def forecast_ml_model(model, last_features, forecast_steps=7, seq_length=30):
    preds = []
    current_seq = last_features.copy()
    for _ in range(forecast_steps):
        pred = model.predict(current_seq.reshape(1, -1))[0]
        preds.append(pred)
        current_seq = np.roll(current_seq, -1)
        current_seq[-1] = pred
    return np.array(preds)

# =============================================================================
# 5. 멀티 모델 예측 및 앙상블 함수
# =============================================================================
def multi_model_forecasting(series, df_prophet, ml_train_X, ml_train_y, ml_model_xgb, ml_model_lgb, lstm_model,
                            forecast_horizon=7, seq_length=30):
    pred_arima = forecast_arima(series, forecast_horizon)
    if prophet_available:
        pred_prophet = forecast_prophet(df_prophet, forecast_horizon)
    else:
        pred_prophet = None
    last_features = ml_train_X[-1]
    pred_xgb = forecast_ml_model(ml_model_xgb, last_features, forecast_steps=forecast_horizon, seq_length=seq_length)
    pred_lgb = forecast_ml_model(ml_model_lgb, last_features, forecast_steps=forecast_horizon, seq_length=seq_length)
    last_seq = ml_train_X[-1].reshape(1, seq_length, -1)
    lstm_model.eval()
    with torch.no_grad():
        last_seq_tensor = torch.FloatTensor(last_seq).to(DEVICE)
        pred_lstm = lstm_model(last_seq_tensor).cpu().numpy().flatten()
        pred_lstm = np.repeat(pred_lstm, forecast_horizon)
    model_preds = {
        'ARIMA': pred_arima.values,
        'XGBoost': pred_xgb,
        'LightGBM': pred_lgb,
        'LSTM': pred_lstm
    }
    if pred_prophet is not None:
        model_preds['Prophet'] = pred_prophet.values
    num_models = len(model_preds)
    weights = {name: 1/num_models for name in model_preds.keys()}
    print("사용된 모델 가중치:", weights)
    ensemble_pred = np.zeros(forecast_horizon)
    for name, pred in model_preds.items():
        ensemble_pred += weights[name] * pred
    for name, pred in model_preds.items():
        print(f"{name} 예측: {pred}")
    return ensemble_pred, model_preds

# =============================================================================
# 6. 유종별 예측 작업 함수 (비동기/병렬 처리용)
# =============================================================================
def forecast_fuel_type(fuel_type, df):
    logger.info(f"예측 시작: {fuel_type}")
    fuel_columns = [col for col in df.columns if col.startswith(f"{fuel_type}_")]
    forecasts = {}
    for target_col in fuel_columns:
        logger.info(f"{target_col} 예측 작업 시작.")
        ts = df.set_index('date')[target_col].dropna()
        if len(ts) < 50:
            logger.warning(f"{target_col}: 데이터 부족하여 건너뜁니다.")
            continue
        train_series = ts[:-7]
        df_prophet = pd.DataFrame({'ds': train_series.index, 'y': train_series.values})
        X_ml, y_ml = [], []
        values = train_series.values
        for i in range(30, len(values)):
            X_ml.append(values[i-30:i].flatten())
            y_ml.append(values[i])
        X_ml = np.array(X_ml)
        y_ml = np.array(y_ml)
        train_size = len(X_ml) - 7
        X_train_ml, y_train_ml = X_ml[:train_size], y_ml[:train_size]
        X_val_ml, y_val_ml = X_ml[train_size:], y_ml[train_size:]
        model_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.01, random_state=RANDOM_SEED)
        model_xgb.fit(X_train_ml, y_train_ml)
        model_lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.01, random_state=RANDOM_SEED)
        model_lgb.fit(X_train_ml, y_train_ml)
        X_train_dl = X_train_ml.reshape(-1, 30, 1)
        X_val_dl = X_val_ml.reshape(-1, 30, 1)
        y_train_dl = torch.FloatTensor(y_train_ml)
        y_val_dl = torch.FloatTensor(y_val_ml)
        lstm_model = train_lstm_model(torch.FloatTensor(X_train_dl), y_train_dl,
                                      torch.FloatTensor(X_val_dl), y_val_dl, input_dim=1, epochs=30)
        ensemble_pred, _ = multi_model_forecasting(train_series, df_prophet, X_ml, y_ml,
                                                   model_xgb, model_lgb, lstm_model,
                                                   forecast_horizon=7, seq_length=30)
        forecast_series = pd.Series(ensemble_pred, index=[ts.index[-1] + timedelta(days=i) for i in range(1,8)])
        logger.info(f"{target_col} 예측 완료.")
        forecasts[target_col] = forecast_series
    return forecasts

# =============================================================================
# 7. 최종 예측: 모든 유종에 대해 일주일치 예측 (병렬 처리)
# =============================================================================
def forecast_all_fuel_types(df):
    fuel_types = ['gasoline', 'diesel', 'premiumGasoline', 'kerosene']
    all_forecasts = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        future_to_ft = {executor.submit(forecast_fuel_type, ft, df): ft for ft in fuel_types}
        for future in concurrent.futures.as_completed(future_to_ft):
            ft = future_to_ft[future]
            try:
                result = future.result()
                all_forecasts[ft] = result
                logger.info(f"{ft} 예측 작업 완료.")
            except Exception as e:
                logger.error(f"{ft} 예측 작업 실패: {e}")
    return all_forecasts

# =============================================================================
# 메인 실행부
# =============================================================================
def main():
    file_path = './korea_economic_data.csv'
    df, numeric_cols, fuel_columns = load_and_preprocess_data(file_path)
    print("\n데이터 일부:")
    print(df.head())
    
    df_features = create_features(df, 'gasoline_National')
    print("\n특성 엔지니어링 완료 후 데이터 일부:")
    print(df_features.head())
    
    all_forecasts = forecast_all_fuel_types(df_features)
    
    # 최종 결과물을 날짜별, 유종별, 지역별로 묶어 JSON으로 출력
    final_result = {}
    for fuel, forecasts in all_forecasts.items():
        final_result[fuel] = {}
        for col, series in forecasts.items():
            # 날짜별로 예측값 저장 (문자열로 변환)
            final_result[fuel][col] = {date.strftime('%Y-%m-%d'): float(val) for date, val in series.items()}
    
    # JSON 최종 결과 출력 및 저장
    final_json = json.dumps(final_result, indent=4)
    print("\n=== 최종 예측 결과 (JSON) ===")
    print(final_json)
    
    # 결과를 파일로 저장
    with open(os.path.join(OUTPUT_DIR, 'final_forecast.json'), 'w') as f:
        f.write(final_json)
    
if __name__ == '__main__':
    main()
