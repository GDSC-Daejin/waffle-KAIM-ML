#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MongoDB 기반 멀티 모델 앙상블 유가 예측 모델
----------------------------------------------------
이 코드는 여러 최신 연구(예: Hyndman & Athanasopoulos, Ensemble Forecast 관련 논문 등)를 참고하여 제작되었습니다.

주요 기법:
    - 데이터 통합: MongoDB에 저장된 국내 경제 지표와 유가 데이터를 통합하여 사용.
      * 시작 날짜(2019-02-20)부터 어제까지의 데이터를 수동으로 조회 (컬렉션 이름은 "Date_YYYY_MM_DD" 형식)
      * ThreadPoolExecutor와 각 Future에 타임아웃을 설정해 한 컬렉션의 지연이 전체를 멈추지 않도록 하고,
        각 컬렉션 조회 전 0.01초 딜레이를 부여.
      * 만약 5초 동안 응답이 없으면 최대 3회까지 재시도합니다.
    - 특성 엔지니어링: 이동평균(MA)와 lag 특성을 생성하여 시계열의 추세와 단기 패턴을 포착.
    - 모델 앙상블: ARIMA, Prophet, XGBoost, LightGBM, LSTM 모델의 예측을 단순 평균 방식으로 앙상블.
    - 비동기(병렬) 처리: ProcessPoolExecutor를 사용하여 4개 유종(휘발유, 경유, 고급휘발유, 등유)의 예측 작업을 동시에 실행.
    - 결과 정리: 최종 예측 결과를 날짜별, 유종별, 지역별로 JSON 파일에 저장.
"""

import os, logging, time, json, warnings, joblib, numpy as np, pandas as pd, re
from datetime import datetime, timedelta
import concurrent.futures

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

# .env 파일 로드 (환경변수 사용)
from dotenv import load_dotenv
load_dotenv()

# MongoDB 연결 설정 (환경변수에서 불러옴)
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

# 로깅 설정 (파일과 콘솔 모두 출력)
def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(levellevelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()  # 콘솔 출력용
    stream_handler.setFormatter(formatter)
    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(level)
    if not logger_obj.handlers:
        logger_obj.addHandler(file_handler)
        logger_obj.addHandler(stream_handler)
    return logger_obj

current_log_folder = os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(current_log_folder, exist_ok=True)
main_logger = setup_logger('main', os.path.join(current_log_folder, 'forecast.log'))

# --- 수동 컬렉션 이름 생성 함수 ---
def generate_collection_names(start_date, end_date):
    """ 시작 날짜부터 종료 날짜까지 "Date_YYYY_MM_DD" 형식의 컬렉션 이름 리스트를 생성 """
    collection_names = []
    current_date = start_date
    while current_date <= end_date:
        collection_names.append(current_date.strftime("Date_%Y_%m_%d"))
        current_date += timedelta(days=1)
    return collection_names

# --- 개별 컬렉션 데이터 조회 함수 (재시도 로직 포함) ---
def load_collection_data(db, coll_name, max_retries=3, retry_delay=5):
    """
    주어진 컬렉션 이름에 대해 데이터를 조회하여 DataFrame으로 변환합니다.
    응답이 없으면 최대 max_retries회 재시도하며, 각 재시도 사이에 retry_delay 초 대기합니다.
    """
    for attempt in range(max_retries):
        try:
            docs = list(db[coll_name].find({}, {"_id": 0}))
            main_logger.info(f"컬렉션 {coll_name}: {len(docs)}개의 도큐먼트 조회 (시도 {attempt+1}/{max_retries})")
            return pd.DataFrame(docs) if docs else None
        except Exception as e:
            main_logger.error(f"컬렉션 {coll_name} 조회 실패 (시도 {attempt+1}/{max_retries}): {e}")
            time.sleep(retry_delay)
    return None

# --- MongoDB 데이터 로드 함수 (수동 컬렉션 이름 사용, 병렬 조회 + 타임아웃 및 재시도 적용) ---
@memory.cache
def load_data_from_mongo(mongo_uri, db_name, start_date, end_date):
    main_logger.info("MongoDB 데이터 로드 시작 (수동 날짜 범위)")
    client = MongoClient(mongo_uri)
    main_logger.info("MongoDB 연결 성공")
    db = client[db_name]
    main_logger.info("데이터베이스 선택 완료")
    
    # 수동으로 컬렉션 이름 생성 (예: 2019-02-20부터 어제까지)
    collection_names = generate_collection_names(start_date, end_date)
    main_logger.info(f"생성된 컬렉션 목록: {collection_names}")
    
    total_docs = 0
    df_list = []
    # ThreadPoolExecutor를 사용하여 병렬로 각 컬렉션을 조회 (각 제출 전에 0.05초 대기)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for coll in collection_names:
            futures[executor.submit(load_collection_data, db, coll)] = coll
        for future in concurrent.futures.as_completed(futures, timeout=60):
            coll = futures[future]
            try:
                df_coll = future.result(timeout=30)
                if df_coll is not None:
                    total_docs += len(df_coll)
                    df_list.append(df_coll)
            except concurrent.futures.TimeoutError:
                main_logger.error(f"컬렉션 {coll} 조회 시간이 초과되었습니다.")
            except Exception as e:
                main_logger.error(f"컬렉션 {coll} 조회 중 오류 발생: {e}")
    main_logger.info(f"전체 도큐먼트 수: {total_docs}")
    client.close()
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = pd.DataFrame()
    main_logger.info(f"MongoDB 데이터 로드 완료: {df.shape}")
    return df

# CSV와 MongoDB 데이터를 통합하여 로드하는 함수
def load_data_from_csv_and_db(mongo_uri, db_name, csv_path):
    """
    CSV 파일에서 기본 데이터를 로드하고, 
    필요한 경우 MongoDB에서 최신 데이터를 추가로 가져옵니다.
    """
    main_logger.info("CSV 및 MongoDB 통합 데이터 로드 시작")
    
    # 1. CSV 파일 로드
    try:
        csv_file = os.path.join(CACHE_DIR, 'korea_economic_data.csv')
        if os.path.exists(csv_file):
            main_logger.info(f"CSV 파일 로드 중: {csv_file}")
            df = pd.read_csv(csv_file)
            
            # date 컬럼이 문자열 'Date_YYYY_MM_DD' 형식이면 변환
            if 'date' in df.columns and isinstance(df['date'].iloc[0], str):
                if df['date'].iloc[0].startswith('Date_'):
                    df['date'] = pd.to_datetime(df['date'].str.replace('Date_', ''), format='%Y_%m_%d')
                else:
                    df['date'] = pd.to_datetime(df['date'])
            
            main_logger.info(f"CSV 파일에서 {len(df)}개 레코드 로드 완료")
            
            # 최신 날짜 확인
            latest_date = df['date'].max()
            main_logger.info(f"CSV 데이터 최신 날짜: {latest_date.strftime('%Y-%m-%d')}")
            
            # 어제 날짜 계산
            yesterday = datetime.today() - timedelta(days=1)
            
            # 최신 데이터가 어제보다 오래된 경우, MongoDB에서 추가 데이터 로드
            if latest_date.date() < yesterday.date():
                main_logger.info(f"최신 데이터 필요: {latest_date.strftime('%Y-%m-%d')} 이후 ~ {yesterday.strftime('%Y-%m-%d')}까지")
                
                # 다음 날부터 어제까지의 데이터 로드
                start_date = latest_date + timedelta(days=1)
                new_data = load_data_from_mongo(mongo_uri, db_name, start_date, yesterday)
                
                if not new_data.empty:
                    main_logger.info(f"MongoDB에서 {len(new_data)}개의 새 레코드를 로드했습니다")
                    
                    # 데이터 통합 전 컬럼 형식 확인 및 조정
                    for col in new_data.columns:
                        if col in df.columns and df[col].dtype != new_data[col].dtype:
                            try:
                                new_data[col] = new_data[col].astype(df[col].dtype)
                            except:
                                main_logger.warning(f"컬럼 {col}의 데이터 형식을 맞추지 못했습니다")
                    
                    # 데이터 통합
                    df = pd.concat([df, new_data], ignore_index=True)
                    
                    # 중복 제거 (혹시 모를 중복을 대비)
                    df = df.drop_duplicates(subset=['date']).reset_index(drop=True)
                    
                    # 통합된 데이터를 CSV에 저장 (최신 상태 유지)
                    backup_path = os.path.join(CACHE_DIR, 'korea_economic_data_backup.csv')
                    if os.path.exists(csv_file):
                        main_logger.info(f"기존 CSV 파일 백업: {backup_path}")
                        try:
                            # 기존 파일 백업
                            import shutil
                            shutil.copy2(csv_file, backup_path)
                        except Exception as e:
                            main_logger.error(f"파일 백업 중 오류: {e}")
                    
                    main_logger.info(f"통합 데이터를 CSV에 저장: {csv_file}")
                    df.to_csv(csv_file, index=False)
                else:
                    main_logger.info("MongoDB에서 새로운 데이터를 찾을 수 없습니다")
            else:
                main_logger.info("CSV 데이터가 최신 상태입니다. MongoDB 조회를 건너뜁니다.")
            
            return df
        else:
            main_logger.error(f"CSV 파일을 찾을 수 없음: {csv_file}")
            # CSV 파일이 없으면 MongoDB 데이터만으로 진행
            return load_data_from_mongo(mongo_uri, db_name, datetime(2019, 2, 20), datetime.today() - timedelta(days=1))
    
    except Exception as e:
        main_logger.error(f"CSV 및 MongoDB 통합 데이터 로드 중 오류: {str(e)}")
        # 오류 발생 시 MongoDB에서 시도
        return load_data_from_mongo(mongo_uri, db_name, datetime(2019, 2, 20), datetime.today() - timedelta(days=1))

def load_and_preprocess_data():
    # DB가 2019-02-20부터 어제까지 데이터가 준비되어 있다고 가정
    start_date = datetime(2019, 2, 20)
    end_date = datetime.today() - timedelta(days=1)
    
    # CSV와 MongoDB 통합 데이터 로드
    df = load_data_from_csv_and_db(MONGO_URI, DB_NAME, 'korea_economic_data.csv')
    
    df['date'] = pd.to_datetime(df['date'].str.replace('Date_', ''), format='%Y_%m_%d')
    df = df.sort_values('date').reset_index(drop=True)
    regions = ['National', 'Seoul', 'Busan', 'Daegu', 'Incheon', 'Gwangju', 'Daejeon', 'Ulsan',
               'Sejong', 'Gyeonggi', 'Gangwon', 'Chungbuk', 'Chungnam', 'Jeonbuk', 'Jeonnam',
               'Gyeongbuk', 'Gyeongnam']
    for fuel_type in ['gasoline', 'premiumGasoline', 'diesel', 'kerosene']:
        if fuel_type in df.columns:
            # 만약 데이터가 문자열이면 eval()로 리스트로 변환
            if isinstance(df[fuel_type].iloc[0], str):
                df[fuel_type] = df[fuel_type].apply(lambda x: eval(x) if isinstance(x, str) else x)
            # 데이터가 리스트이면 내부의 문자열을 float으로 변환
            elif isinstance(df[fuel_type].iloc[0], list):
                df[fuel_type] = df[fuel_type].apply(lambda x: [float(item) for item in x] if isinstance(x, list) else x)
            # 리스트의 길이가 지역 수와 동일하면 각 지역별 컬럼 생성
            if isinstance(df[fuel_type].iloc[0], list) and len(df[fuel_type].iloc[0]) == len(regions):
                for i, region in enumerate(regions):
                    df[f'{fuel_type}_{region}'] = df[fuel_type].apply(lambda x: float(x[i]) if isinstance(x, list) else np.nan)
            df = df.drop(fuel_type, axis=1)
    if 'area' in df.columns and isinstance(df['area'].iloc[0], str):
        df = df.drop('area', axis=1)
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    fuel_types = ['gasoline', 'premiumGasoline', 'diesel', 'kerosene']
    fuel_columns = {ft: [col for col in df.columns if col.startswith(f"{ft}_")] for ft in fuel_types}
    main_logger.info(f"예측 대상 변수: {fuel_columns}")
    main_logger.info(f"데이터 기간: {df['date'].min().strftime('%Y-%m-%d')} 부터 {df['date'].max().strftime('%Y-%m-%d')}, 총 {df.shape[0]}일")
    return df, numeric_cols, fuel_columns

# =============================================================================
# 2. 특성 엔지니어링
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
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    for feature in [target_col]:
        for window in [3, 7, 14]:
            df[f'{feature}_MA{window}'] = df[target_col].rolling(window=window).mean()
        for lag in [1, 2, 3]:
            df[f'{feature}_lag{lag}'] = df[target_col].shift(lag)
    df = df.dropna()
    main_logger.info(f"특성 생성 후 데이터 크기: {df.shape}")
    main_logger.info("=== 특성 엔지니어링 완료 ===")
    return df

# =============================================================================
# 3. 딥러닝 모델 (LSTM) 정의 및 학습
# =============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
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
    main_logger.info(f"예측 시작: {fuel_type}")
    fuel_columns = [col for col in df.columns if col.startswith(f"{fuel_type}_")]
    forecasts = {}
    for target_col in fuel_columns:
        main_logger.info(f"{target_col} 예측 작업 시작.")
        ts = df.set_index('date')[target_col].dropna()
        if len(ts) < 50:
            main_logger.warning(f"{target_col}: 데이터 부족하여 건너뜁니다.")
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
        forecast_series = pd.Series(ensemble_pred, index=[ts.index[-1] + timedelta(days=i) for i in range(1, 8)])
        main_logger.info(f"{target_col} 예측 완료.")
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
                main_logger.info(f"{ft} 예측 작업 완료.")
            except Exception as e:
                main_logger.error(f"{ft} 예측 작업 실패: {e}")
    return all_forecasts

# =============================================================================
# 메인 실행부
# =============================================================================
def main():
    df, numeric_cols, fuel_columns = load_and_preprocess_data()
    print("\n데이터 일부:")
    print(df.head())
    
    df_features = create_features(df, 'gasoline_National')
    print("\n특성 엔지니어링 완료 후 데이터 일부:")
    print(df_features.head())
    
    all_forecasts = forecast_all_fuel_types(df_features)
    
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
