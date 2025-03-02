#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MongoDB 기반 멀티 모델 앙상블 유가 예측 모델
----------------------------------------------------
이 코드는 여러 최신 연구 및 논문(예: Hyndman & Athanasopoulos, ensemble forecast 연구 등)을 
참고하여 제작되었습니다.
주요 기법:
    - **데이터 통합:** MongoDB에 저장된 국내 경제 지표와 유가 데이터를 통합하여 사용.
    - **특성 엔지니어링:** 이동평균, lag, 차분 등 시계열 기법을 활용하여 예측에 유용한 특성 생성.
      (참고: "Forecasting: Principles and Practice")
    - **모델 앙상블:** ARIMA, Prophet, XGBoost, LightGBM, LSTM 모델의 예측을 앙상블하여 예측 성능 개선.
      (참고: Ensemble Learning 관련 연구)
    - **비동기(병렬) 처리:** ProcessPoolExecutor를 사용하여 각 유종별 예측 작업을 병렬로 실행.
    - **결과 정리:** 예측 결과를 날짜별, 유종별, 지역별로 JSON 파일에 저장.
"""

import os, logging, time, json, warnings, joblib, numpy as np, pandas as pd
from datetime import datetime, timedelta
import concurrent.futures

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv()

# MongoDB 관련 라이브러리
from pymongo import MongoClient

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
LOG_DIR = './logs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# joblib 캐시 설정 (재실행 시 속도 향상)
memory = joblib.Memory(location=CACHE_DIR, verbose=0)

# MongoDB 연결 설정
# 환경변수에서 DB 설정 로드
MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = os.environ.get("DB_NAME")

# 시드 설정 (재현성을 위해)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {DEVICE}")

# 로깅 설정 (각 작업별 로그는 /logs/현재날짜_시간 폴더에 저장)
def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler  = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(level)
    if not logger_obj.handlers:
        logger_obj.addHandler(handler)
    return logger_obj

# 현재 시각 기반 로그 폴더 생성
current_log_folder = os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(current_log_folder, exist_ok=True)
main_logger = setup_logger('main', os.path.join(current_log_folder, 'forecast.log'))

# =============================================================================
# 1. MongoDB 데이터 로드 및 전처리 (캐싱 적용)
# =============================================================================
@memory.cache
def load_data_from_mongo(mongo_uri, db_name):
    main_logger.info("MongoDB 데이터 로드 시작")
    client = MongoClient(mongo_uri)
    db = client[db_name]
    # 컬렉션 이름이 "Date_"로 시작하는 모든 컬렉션을 불러옴
    coll_names = [name for name in db.list_collection_names() if name.startswith("Date_")]
    main_logger.info(f"발견된 컬렉션 수: {len(coll_names)}")
    df_list = []
    for coll in coll_names:
        docs = list(db[coll].find({}, {"_id":0}))
        if docs:
            temp_df = pd.DataFrame(docs)
            df_list.append(temp_df)
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = pd.DataFrame()
    client.close()
    main_logger.info(f"MongoDB 데이터 로드 완료: {df.shape}")
    return df

def load_and_preprocess_data():
    # MongoDB에서 데이터 로드
    df = load_data_from_mongo(MONGO_URI, DB_NAME)
    # 날짜 처리: "Date_" 접두어 제거 후 정렬
    df['date'] = pd.to_datetime(df['date'].str.replace('Date_', ''), format='%Y_%m_%d')
    df = df.sort_values('date').reset_index(drop=True)
    # 지역별 연료 가격 데이터 처리: 문자열 리스트 -> 수치형 리스트 변환
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
    # 자료형 최적화: float64 -> float32
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # 유종별 열 추출: 각 유종에 대해 전국 및 지역별 데이터 리스트
    fuel_types = ['gasoline', 'premiumGasoline', 'diesel', 'kerosene']
    fuel_columns = {}
    for ft in fuel_types:
        fuel_columns[ft] = [col for col in df.columns if col.startswith(f"{ft}_")]
    main_logger.info(f"예측 대상 변수: {fuel_columns}")
    main_logger.info(f"데이터 기간: {df['date'].min().strftime('%Y-%m-%d')} 부터 {df['date'].max().strftime('%Y-%m-%d')}, 총 {df.shape[0]}일")
    return df, numeric_cols, fuel_columns

# =============================================================================
# 2. 특성 엔지니어링
# (특성 엔지니어링: 원시 데이터를 이동평균, lag, 차분 등으로 변환해 예측에 유리한 형식으로 만듦)
# =============================================================================
@memory.cache
def create_features(df, target_col):
    main_logger.info("=== 특성 엔지니어링 시작 ===")
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    # 이동평균 (MA)와 lag 특성은 시계열 데이터의 추세와 계절성을 포착하는 데 유용함.
    for feature in [target_col]:
        for window in [3,7,14]:
            df[f'{feature}_MA{window}'] = df[target_col].rolling(window=window).mean()
        for lag in [1,2,3]:
            df[f'{feature}_lag{lag}'] = df[target_col].shift(lag)
    df = df.dropna()
    main_logger.info(f"특성 생성 후 데이터 크기: {df.shape}")
    main_logger.info("=== 특성 엔지니어링 완료 ===")
    return df

# =============================================================================
# 3. 딥러닝 모델 (LSTM) 정의 및 학습
# (LSTM: 순환신경망(RNN)의 한 종류로, 장기 의존성을 학습하는 데 효과적임)
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
# (ARIMA: 통계적 시계열 모델 / Prophet: Facebook Prophet, XGBoost/LightGBM: 부스팅 계열, LSTM: 딥러닝)
# =============================================================================
def forecast_arima(series, forecast_horizon=7, order=(5,1,0), seasonal_order=(0,0,0,0)):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    pred = fitted.get_forecast(steps=forecast_horizon)
    return pred.predicted_mean

def get_korean_holidays(years):
    # 사용자 정의 공휴일 함수 (한국의 주요 공휴일을 직접 정의할 수 있음)
    holidays = []
    for year in years:
        holidays.append({'holiday': 'New Year', 'ds': f'{year}-01-01', 'lower_window': 0, 'upper_window': 1})
        # 추가: 설날, 추석, 광복절 등 필요 시 추가
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
# (각 모델의 예측값을 동일 가중치로 앙상블하는 단순 평균 기법 사용)
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
    # 동일 가중치로 앙상블 (추후 가중치 최적화 기법 적용 가능)
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
# (각 유종별로 전국 및 지역별 예측 작업을 수행하며, MongoDB 데이터 및 경제 지표 등 
#  모든 데이터를 활용하여 보다 정밀한 예측을 시도)
# =============================================================================
def forecast_fuel_type(fuel_type, df):
    logger.info(f"예측 시작: {fuel_type}")
    # 해당 유종의 모든 열(전국 및 지역별)
    fuel_columns = [col for col in df.columns if col.startswith(f"{fuel_type}_")]
    forecasts = {}
    for target_col in fuel_columns:
        logger.info(f"{target_col} 예측 작업 시작.")
        ts = df.set_index('date')[target_col].dropna()
        if len(ts) < 50:
            logger.warning(f"{target_col}: 데이터 부족하여 건너뜁니다.")
            continue
        # 데이터 분할: 마지막 7일은 검증/예측을 위해 제외
        train_series = ts[:-7]
        # Prophet 모델용 데이터 준비 (경제 지표 등 추가 가능)
        df_prophet = pd.DataFrame({'ds': train_series.index, 'y': train_series.values})
        # 특성 엔지니어링: 여기서는 30일 window 이동평균 및 lag를 생성
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
        # 머신러닝 모델 학습 (XGBoost, LightGBM)
        model_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.01, random_state=RANDOM_SEED)
        model_xgb.fit(X_train_ml, y_train_ml)
        model_lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.01, random_state=RANDOM_SEED)
        model_lgb.fit(X_train_ml, y_train_ml)
        # 딥러닝 모델 학습 (LSTM)
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
# (ProcessPoolExecutor를 사용하여 4개 유종의 예측 작업을 동시에 수행)
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
    # MongoDB에서 데이터를 불러오고 전처리
    df, numeric_cols, fuel_columns = load_and_preprocess_data()
    print("\n데이터 일부:")
    print(df.head())
    
    # 예를 들어, gasoline_National을 대상으로 특성 엔지니어링 수행
    df_features = create_features(df, 'gasoline_National')
    print("\n특성 엔지니어링 완료 후 데이터 일부:")
    print(df_features.head())
    
    # 병렬(비동기) 처리로 모든 유종 예측 실행
    all_forecasts = forecast_all_fuel_types(df_features)
    
    # 최종 결과물을 날짜별, 유종별, 지역별로 묶어 JSON으로 출력
    final_result = {}
    for fuel, forecasts in all_forecasts.items():
        final_result[fuel] = {}
        for col, series in forecasts.items():
            final_result[fuel][col] = {date.strftime('%Y-%m-%d'): float(val) for date, val in series.items()}
    
    final_json = json.dumps(final_result, indent=4)
    print("\n=== 최종 예측 결과 (JSON) ===")
    print(final_json)
    
    with open(os.path.join(OUTPUT_DIR, 'final_forecast.json'), 'w') as f:
        f.write(final_json)
    
if __name__ == '__main__':
    main()
