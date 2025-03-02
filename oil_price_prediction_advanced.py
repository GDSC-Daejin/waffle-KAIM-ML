"""
고급 유가 예측 모델 - 다중 모델 앙상블 및 변수 중요도 분석

이 스크립트는 한국 경제 데이터와 국제 유가 데이터를 사용하여 국내 유가를 예측합니다.
여러 시계열 모델(LSTM, GRU, ARIMA, Prophet, XGBoost, LightGBM)을 앙상블하여 
더 정확한 예측을 수행하고, 변수 중요도를 분석합니다.

주요 기능:
1. 데이터 로드 및 전처리
2. 탐색적 데이터 분석
3. 변수 중요도 분석
4. 다양한 시계열 모델 학습
5. 앙상블 모델 구축
6. 예측 결과 시각화

이 모델은 정확도를 최우선으로 하며, 일별 예측을 수행합니다.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import joblib
import gc
import time
import json
from tqdm import tqdm

# 모델링 관련 라이브러리
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
import statsmodels.api as sm

# Prophet 라이브러리 임포트 시도
try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    print("Prophet 라이브러리가 설치되어 있지 않습니다.")
    prophet_available = False

# Hyperopt 라이브러리 임포트 시도 (하이퍼파라미터 최적화)
try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    hyperopt_available = True
except ImportError:
    print("hyperopt 라이브러리가 설치되어 있지 않습니다. 수동 튜닝을 사용합니다.")
    hyperopt_available = False

# 경고 필터링
warnings.filterwarnings('ignore')

# 결과 저장 디렉토리 설정
OUTPUT_DIR = '/Users/yangjunseok/Documents/CodeSpace/KAIM-ML/waffle-KAIM-ML/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 시드 설정
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

# GPU 사용 가능 여부 확인
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {DEVICE}")

# =============================================================================
# 1. 데이터 로드 및 전처리
# =============================================================================

def load_and_preprocess_data(file_path):
    """
    데이터를 로드하고 전처리하는 함수
    
    Args:
        file_path (str): CSV 파일 경로
        
    Returns:
        tuple: (전처리된 데이터프레임, 수치형 열 목록, 타겟 컬럼명)
    """
    print(f"데이터 로드 중: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"원본 데이터 크기: {df.shape}")
    
    # 날짜 열 처리
    df['date'] = pd.to_datetime(df['date'].str.replace('Date_', ''), format='%Y_%m_%d')
    df = df.sort_values('date').reset_index(drop=True)
    
    # 지역별 연료 가격 데이터 처리 (문자열 형태의 리스트를 실제 값으로 변환)
    regions = ['National', 'Seoul', 'Busan', 'Daegu', 'Incheon', 'Gwangju', 'Daejeon', 'Ulsan', 
               'Sejong', 'Gyeonggi', 'Gangwon', 'Chungbuk', 'Chungnam', 'Jeonbuk', 'Jeonnam', 
               'Gyeongbuk', 'Gyeongnam']
    
    # 각 연료 유형(휘발유, 고급휘발유, 경유, 등유) 처리
    for fuel_type in ['gasoline', 'premiumGasoline', 'diesel', 'kerosene']:
        if fuel_type in df.columns and isinstance(df[fuel_type].iloc[0], str):
            # 문자열을 리스트로 변환
            df[fuel_type] = df[fuel_type].apply(lambda x: eval(x) if isinstance(x, str) else x)
            
            # 각 지역별 열 생성
            if isinstance(df[fuel_type].iloc[0], list) and len(df[fuel_type].iloc[0]) == len(regions):
                for i, region in enumerate(regions):
                    df[f'{fuel_type}_{region}'] = df[fuel_type].apply(lambda x: float(x[i]) if isinstance(x, list) else np.nan)
            
            # 원래 컬럼 삭제
            df = df.drop(fuel_type, axis=1)
    
    # 'area' 컬럼 삭제 (이미 지역별로 분리했으므로)
    if 'area' in df.columns and isinstance(df['area'].iloc[0], str):
        df = df.drop('area', axis=1)
    
    # 결측치 처리
    print(f"결측치 개수: {df.isnull().sum().sum()}")
    
    # 데이터 유형 최적화
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
            
    # 수치형 열 식별
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # 타겟 컬럼 설정 (국내 전체 휘발유 가격)
    target_col = 'gasoline_National'
    
    print(f"전처리 후 데이터 크기: {df.shape}, 수치형 변수 수: {len(numeric_cols)}")
    
    # 기본 정보 출력
    print("\n데이터 기간:", df['date'].min().strftime('%Y-%m-%d'), "부터", 
          df['date'].max().strftime('%Y-%m-%d'), "까지")
    print(f"총 {df.shape[0]}일 데이터")
    
    return df, numeric_cols, target_col

# =============================================================================
# 2. 탐색적 데이터 분석 (EDA)
# =============================================================================

def exploratory_data_analysis(df, numeric_cols, target_col):
    """
    탐색적 데이터 분석을 수행하고 결과 시각화
    
    Args:
        df: 데이터프레임
        numeric_cols: 수치형 열 목록
        target_col: 예측할 타겟 컬럼명
        
    Returns:
        tuple: (상관관계 행렬, 주요 특성 목록)
    """
    print("\n=== 탐색적 데이터 분석(EDA) 시작 ===")
    
    # 기초 통계량
    print("\n기본 통계량:")
    print(df[numeric_cols].describe().T)
    
    # 타겟 변수(휘발유 가격) 분포 시각화
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df[target_col], kde=True)
    plt.title(f'{target_col} 분포')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[target_col])
    plt.title(f'{target_col} 박스플롯')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'target_distribution.png'))
    plt.close()
    
    # 시계열 추세 시각화
    plt.figure(figsize=(15, 8))
    plt.plot(df['date'], df[target_col], label=target_col)
    
    # 국제 유가 추가
    for oil in ['Dubai_Val', 'Brent_Val', 'WTI_Val']:
        if (oil in df.columns) and (df[oil].dtype != 'object'):
            plt.plot(df['date'], df[oil], label=oil, alpha=0.7)
    
    plt.title('유가 추세')
    plt.xlabel('날짜')
    plt.ylabel('가격')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'oil_price_trend.png'))
    plt.close()
    
    # 상관관계 분석
    corr_matrix = df[numeric_cols].corr()
    
    # 타겟 변수와의 상관관계가 높은 상위 변수들
    target_correlations = corr_matrix[target_col].sort_values(ascending=False)
    print("\n타겟 변수와 상관관계가 높은 특성들:")
    print(target_correlations.head(10))
    
    # 상관관계 히트맵 (타겟과 관련된 상위 특성들만)
    top_features = target_correlations[1:11].index.tolist()  # 타겟 자신을 제외한 상위 10개
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix.loc[top_features + [target_col], top_features + [target_col]], 
                annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('주요 변수들 간 상관관계')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'))
    plt.close()
    
    # 주요 변수들과 타겟 간의 산점도
    plt.figure(figsize=(15, 12))
    for i, feature in enumerate(top_features[:9]):  # 상위 9개 특성만 표시
        plt.subplot(3, 3, i+1)
        sns.scatterplot(x=df[feature], y=df[target_col])
        plt.title(f'{feature} vs {target_col}')
        plt.xlabel(feature)
        plt.ylabel(target_col)
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_scatter_plots.png'))
    plt.close()
    
    # 시계열 자기 상관 (ACF) 및 부분 자기 상관 (PACF) 분석
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    sm.graphics.tsa.plot_acf(df[target_col].dropna(), lags=50, ax=plt.gca())
    plt.subplot(212)
    sm.graphics.tsa.plot_pacf(df[target_col].dropna(), lags=50, ax=plt.gca())
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'acf_pacf_analysis.png'))
    plt.close()
    
    # 정상성 검정 (Augmented Dickey-Fuller test)
    print("\n타겟 변수 정상성 검정 (ADF):")
    adf_result = adfuller(df[target_col].dropna())
    print(f'ADF 검정통계량: {adf_result[0]}')
    print(f'p-값: {adf_result[1]}')
    print(f'결론: {"정상 시계열" if adf_result[1] < 0.05 else "비정상 시계열 (차분 필요)"}')
    
    # 1차 차분 시계열 시각화 및 검정
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(df['date'][1:], np.diff(df[target_col]))
    plt.title(f'{target_col} 1차 차분')
    plt.grid(True)
    
    # 2차 차분 시계열
    plt.subplot(2, 1, 2)
    plt.plot(df['date'][2:], np.diff(np.diff(df[target_col])))
    plt.title(f'{target_col} 2차 차분')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'differenced_series.png'))
    plt.close()
    
    # 1차 차분 정상성 검정
    diff1 = np.diff(df[target_col].dropna())
    adf_diff1 = adfuller(diff1)
    print(f'\n1차 차분 ADF 검정통계량: {adf_diff1[0]}')
    print(f'p-값: {adf_diff1[1]}')
    print(f'결론: {"정상 시계열" if adf_diff1[1] < 0.05 else "비정상 시계열 (추가 차분 필요)"}')
    
    # 계절성 검사 (연간 주기 또는 주간 주기)
    plt.figure(figsize=(15, 7))
    pd.Series(df[target_col]).plot(title=f"{target_col} 시계열 - 계절성 확인")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'seasonality_check.png'))
    plt.close()
    
    print("\n=== 탐색적 데이터 분석(EDA) 완료 ===")
    
    return corr_matrix, top_features

# =============================================================================
# 3. 특성 엔지니어링 및 시계열 데이터 준비
# =============================================================================

def create_features(df, target_col):
    """
    예측에 도움이 되는 추가 특성을 생성합니다.
    
    Args:
        df: 원본 데이터프레임
        target_col: 타겟 변수 이름
    
    Returns:
        df: 특성이 추가된 데이터프레임
    """
    print("\n=== 특성 엔지니어링 시작 ===")
    
    # 작업용 데이터프레임 복사
    df = df.copy()
    
    # 날짜 관련 특성
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=월요일, 6=일요일
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 주요 경제/유가 변수의 이동평균 및 추세 특성
    key_features = ['Dubai_Val', 'Brent_Val', 'WTI_Val', 'KRW_Rating', target_col]
    key_features = [f for f in key_features if f in df.columns]
    
    for feature in key_features:
        # 이동평균 (다양한 창 크기)
        for window in [3, 7, 14, 30]:
            df[f'{feature}_MA{window}'] = df[feature].rolling(window=window).mean()
        
        # 지수이동평균
        for span in [7, 14, 30]:
            df[f'{feature}_EMA{span}'] = df[feature].ewm(span=span).mean()
        
        # 표준편차 (변동성)
        df[f'{feature}_std7'] = df[feature].rolling(window=7).std()
        
        # 모멘텀/추세 지표 (변화율)
        for shift in [1, 3, 7, 14]:
            df[f'{feature}_change{shift}'] = df[feature].pct_change(periods=shift)
        
        # 지연 특성 (lag features)
        for lag in [1, 2, 3, 7]:
            df[f'{feature}_lag{lag}'] = df[feature].shift(lag)
    
    # 국제 유가와 환율 간의 교호작용 특성
    if 'KRW_Rating' in df.columns:
        for oil in ['Dubai_Val', 'Brent_Val', 'WTI_Val']:
            if oil in df.columns:
                df[f'{oil}_KRW_ratio'] = df[oil] / df['KRW_Rating']
                df[f'{oil}_KRW_product'] = df[oil] * df['KRW_Rating']
    
    # 국내 휘발유 가격과 국제 유가의 가격 차이 (시차 고려)
    if target_col in df.columns and 'Dubai_Val' in df.columns:
        df['price_diff_dubai'] = df[target_col] - df['Dubai_Val']
        for lag in [1, 3, 7]:
            df[f'price_diff_dubai_lag{lag}'] = df['price_diff_dubai'].shift(lag)
    
    # 계절성 특성 (사인/코사인 변환)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # 결측치 처리
    print(f"특성 생성 후 결측치 개수: {df.isnull().sum().sum()}")
    
    # 처음 몇 행의 결측치는 단순히 제거 (이동평균 등으로 인한 결측치)
    df = df.dropna()
    
    print(f"특성 엔지니어링 후 데이터 크기: {df.shape}")
    print("\n=== 특성 엔지니어링 완료 ===")
    
    return df

def prepare_time_series_data(df, target_col, numeric_cols, seq_length=30, test_ratio=0.15):
    """
    시계열 예측을 위한 데이터셋을 준비합니다.
    
    Args:
        df: 전처리된 데이터프레임
        target_col: 예측 대상 컬럼
        numeric_cols: 수치형 데이터 컬럼 목록
        seq_length: 시퀀스 길이 (과거 몇 일의 데이터를 사용할지)
        test_ratio: 테스트 세트의 비율
    
    Returns:
        dict: 학습에 필요한 데이터셋 및 관련 정보
    """
    print("\n=== 시계열 데이터 준비 시작 ===")
    
    # 불필요한 컬럼 제거
    features_df = df[numeric_cols].copy()
    dates = df['date']
    
    # 데이터 스케일링
    scalers = {}
    scaled_data = {}
    
    # 타겟 변수 스케일링
    scalers['target'] = MinMaxScaler()
    scaled_data['target'] = scalers['target'].fit_transform(features_df[[target_col]]).flatten()
    
    # 특성 변수 스케일링
    feature_cols = [col for col in features_df.columns if col != target_col]
    scalers['features'] = StandardScaler()
    scaled_data['features'] = scalers['features'].fit_transform(features_df[feature_cols])
    
    # 훈련/테스트 분할
    test_size = int(len(df) * test_ratio)
    train_size = len(df) - test_size
    
    print(f"훈련 데이터 기간: {dates.iloc[0]} ~ {dates.iloc[train_size-1]}")
    print(f"테스트 데이터 기간: {dates.iloc[train_size]} ~ {dates.iloc[-1]}")
    
    # 1. 딥러닝 모델용 시퀀스 데이터
    X_seq, y_seq = [], []
    for i in range(seq_length, len(features_df)):
        # 모든 특성을 포함한 시퀀스 생성
        features_seq = np.column_stack((
            scaled_data['features'][i-seq_length:i],    # 모든 특성
            scaled_data['target'][i-seq_length:i].reshape(-1, 1)  # 타겟 변수의 과거 값
        ))
        X_seq.append(features_seq)
        y_seq.append(scaled_data['target'][i])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # 훈련/테스트 분할
    train_end_idx = train_size - seq_length
    
    X_train_seq = X_seq[:train_end_idx]
    y_train_seq = y_seq[:train_end_idx]
    X_test_seq = X_seq[train_end_idx:]
    y_test_seq = y_seq[train_end_idx:]
    
    # PyTorch 텐서로 변환
    X_train_tensor = torch.FloatTensor(X_train_seq)
    y_train_tensor = torch.FloatTensor(y_train_seq)
    X_test_tensor = torch.FloatTensor(X_test_seq)
    y_test_tensor = torch.FloatTensor(y_test_seq)
    
    # 2. 전통적인 시계열 모델용 데이터
    train_series = scaled_data['target'][:train_size]
    test_series = scaled_data['target'][train_size:]
    
    # 3. 머신러닝 모델용 특성 행렬
    # 과거 시퀀스 기반으로 특성 생성 (평탄화)
    X_ml = []
    for i in range(seq_length, len(features_df)):
        # 특성 시퀀스를 1차원으로 평탄화
        ml_features = np.concatenate([
            scaled_data['features'][i-seq_length:i].flatten(),
            scaled_data['target'][i-seq_length:i]
        ])
        X_ml.append(ml_features)
    
    X_ml = np.array(X_ml)
    y_ml = y_seq
    
    X_train_ml = X_ml[:train_end_idx]
    y_train_ml = y_ml[:train_end_idx]
    X_test_ml = X_ml[train_end_idx:]
    y_test_ml = y_ml[train_end_idx:]
    
    # 4. Prophet 모델용 데이터
    prophet_train = pd.DataFrame({
        'ds': dates[:train_size],
        'y': features_df[target_col].iloc[:train_size]
    })
    
    prophet_test = pd.DataFrame({
        'ds': dates[train_size:],
        'y': features_df[target_col].iloc[train_size:]
    })
    
    # 테스트 데이터 날짜 (예측 평가용)
    test_dates = dates[train_size:].reset_index(drop=True)
    
    # 컬럼 이름 저장
    feature_names = feature_cols
    
    print(f"시퀀스 데이터 준비 완료: 입력 형태 {X_train_seq.shape}, 타겟 형태 {y_train_seq.shape}")
    print(f"테스트 데이터 크기: {X_test_seq.shape[0]} 샘플")
    print("\n=== 시계열 데이터 준비 완료 ===")
    
    return {
        'deep_learning': {
            'X_train': X_train_tensor,
            'y_train': y_train_tensor,
            'X_test': X_test_tensor,
            'y_test': y_test_tensor
        },
        'ml': {
            'X_train': X_train_ml,
            'y_train': y_train_ml,
            'X_test': X_test_ml,
            'y_test': y_test_ml
        },
        'time_series': {
            'train': train_series,
            'test': test_series
        },
        'prophet': {
            'train': prophet_train,
            'test': prophet_test
        },
        'dates': {
            'train_dates': dates[:train_size],
            'test_dates': test_dates
        },
        'original': {
            'target_values': features_df[target_col],
            'target_train': features_df[target_col].iloc[:train_size],
            'target_test': features_df[target_col].iloc[train_size:],
        },
        'metadata': {
            'seq_length': seq_length,
            'feature_names': feature_names,
            'target_col': target_col
        },
        'scalers': scalers
    }

# =============================================================================
# 4. 변수 중요도 분석
# =============================================================================

def analyze_feature_importance(df, target_col, numeric_cols, top_features=None):
    """
    변수 중요도를 분석하여 예측에 가장 영향을 많이 미치는 특성들을 파악합니다.
    
    Args:
        df: 데이터프레임
        target_col: 타겟 변수명
        numeric_cols: 수치형 특성 목록
        top_features: 상관관계 분석에서 얻은 상위 특성들 (옵션)
        
    Returns:
        dict: 변수 중요도 결과
    """
    print("\n=== 변수 중요도 분석 시작 ===")
    
    # 분석에 사용할 특성 선택 (결측치 없는 열만)
    features = [col for col in numeric_cols if col != target_col]
    data_for_analysis = df[features + [target_col]].dropna()
    
    X = data_for_analysis[features]
    y = data_for_analysis[target_col]
    
    # 1. 랜덤 포레스트 기반 특성 중요도
    print("랜덤 포레스트 변수 중요도 계산 중...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf_model.fit(X, y)
    
    # 변수 중요도 저장
    rf_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("상위 10개 중요 변수 (랜덤 포레스트):")
    print(rf_importance.head(10))
    
    # 2. 순열 중요도 (더 신뢰성 있는 중요도 측정 방법)
    print("\n순열 중요도 계산 중... (시간이 다소 소요될 수 있습니다)")
    
    # 계산 시간 단축을 위해 샘플링
    sample_size = min(5000, len(X))
    random_indices = np.random.choice(len(X), size=sample_size, replace=False)
    X_sample = X.iloc[random_indices]
    y_sample = y.iloc[random_indices]
    
    # 시간을 많이 소요하는 순열 중요도 계산
    # 빠른 계산을 위해 XGBoost 모델 사용
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=RANDOM_SEED)
    xgb_model.fit(X_sample, y_sample)
    
    # 순열 중요도 계산 (n_repeats=2로 시간 단축)
    perm_importance = permutation_importance(
        xgb_model, X_sample, y_sample, 
        n_repeats=2, random_state=RANDOM_SEED, n_jobs=-1
    )
    
    # 중요도 결과 저장
    perm_importance_result = pd.DataFrame({
        'Feature': features,
        'Importance': perm_importance.importances_mean
    }).sort_values('Importance', ascending=False)
    
    print("상위 10개 중요 변수 (순열 중요도):")
    print(perm_importance_result.head(10))
    
    # 3. XGBoost feature importance
    print("\nXGBoost 변수 중요도 계산 중...")
    
    # 전체 데이터로 XGBoost 학습
    xgb_full = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=RANDOM_SEED)
    xgb_full.fit(X, y)
    
    # XGBoost 특성 중요도
    xgb_importance = pd.DataFrame({
        'Feature': features,
        'Importance': xgb_full.feature_importances_,
    }).sort_values('Importance', ascending=False)
    
    print("상위 10개 중요 변수 (XGBoost):")
    print(xgb_importance.head(10))
    
    # 변수 중요도 시각화
    plt.figure(figsize=(12, 8))
    
    # 상위 20개 특성만 표시
    top_n = 20
    top_features_rf = rf_importance.head(top_n)
    
    sns.barplot(x='Importance', y='Feature', data=top_features_rf)
    plt.title('상위 20개 변수 중요도 (랜덤 포레스트)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_rf.png'))
    plt.close()
    
    # 순열 중요도 시각화
    plt.figure(figsize=(12, 8))
    top_features_perm = perm_importance_result.head(top_n)
    sns.barplot(x='Importance', y='Feature', data=top_features_perm)
    plt.title('상위 20개 변수 중요도 (순열 중요도)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_permutation.png'))
    plt.close()
    
    # XGBoost 중요도 시각화
    plt.figure(figsize=(12, 8))
    top_features_xgb = xgb_importance.head(top_n)
    sns.barplot(x='Importance', y='Feature', data=top_features_xgb)
    plt.title('상위 20개 변수 중요도 (XGBoost)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_xgb.png'))
    plt.close()
    
    # 종합 변수 중요도 계산 (세 방식의 평균 순위)
    # 각 방법별 순위 계산
    rf_ranks = rf_importance.reset_index().reset_index().rename(columns={'level_0': 'RF_Rank'})
    perm_ranks = perm_importance_result.reset_index().reset_index().rename(columns={'level_0': 'Perm_Rank'})
    xgb_ranks = xgb_importance.reset_index().reset_index().rename(columns={'level_0': 'XGB_Rank'})
    
    # 순위 정보 통합
    combined_ranks = rf_ranks[['Feature', 'RF_Rank']]\
        .merge(perm_ranks[['Feature', 'Perm_Rank']], on='Feature')\
        .merge(xgb_ranks[['Feature', 'XGB_Rank']], on='Feature')
    
    # 평균 순위 계산
    combined_ranks['Avg_Rank'] = (combined_ranks['RF_Rank'] + combined_ranks['Perm_Rank'] + combined_ranks['XGB_Rank']) / 3
    combined_ranks = combined_ranks.sort_values('Avg_Rank')
    
    print("\n종합 변수 중요도 (세 방법의 순위 평균):")
    print(combined_ranks[['Feature', 'Avg_Rank']].head(10))
    
    # 결과 저장
    importance_results = {
        'random_forest': rf_importance,
        'permutation': perm_importance_result,
        'xgboost': xgb_importance,
        'combined': combined_ranks
    }
    
    # CSV로 저장
    combined_ranks.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)
    
    print("\n=== 변수 중요도 분석 완료 ===")
    
    return importance_results, combined_ranks['Feature'].iloc[:30].tolist()  # 상위 30개 특성 반환

# =============================================================================
# 5. 딥러닝 모델 정의
# =============================================================================

class LSTMModel(nn.Module):
    """LSTM 기반 시계열 예측 모델"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x):
        # LSTM 레이어
        lstm_out, _ = self.lstm(x)
        
        # 마지막 시간 단계의 출력만 사용
        last_time_step = lstm_out[:, -1, :]
        
        # 출력층
        out = self.fc1(last_time_step)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class GRUModel(nn.Module):
    """GRU 기반 시계열 예측 모델"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super(GRUModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x):
        # GRU 레이어
        gru_out, _ = self.gru(x)
        
        # 마지막 시간 단계의 출력만 사용
        last_time_step = gru_out[:, -1, :]
        
        # 출력층
        out = self.fc1(last_time_step)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class BiLSTMModel(nn.Module):
    """양방향 LSTM 기반 시계열 예측 모델"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 양방향 LSTM은 양쪽 방향의 출력을 합치므로 hidden_dim이 2배
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # 양방향 LSTM 레이어
        lstm_out, _ = self.lstm(x)
        
        # 마지막 시간 단계의 출력만 사용
        last_time_step = lstm_out[:, -1, :]
        
        # 출력층
        out = self.fc1(last_time_step)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# =============================================================================
# 6. 모델 학습 및 평가 함수
# =============================================================================

def train_deep_learning_model(model, train_data, test_data, model_name, 
                              epochs=100, batch_size=32, patience=10, lr=0.001):
    """
    딥러닝 모델 학습 및 평가 함수
    
    Args:
        model: 학습할 PyTorch 모델
        train_data: 훈련 데이터 (X_train, y_train)
        test_data: 테스트 데이터 (X_test, y_test)
        model_name: 모델 이름 (저장 및 로깅용)
        epochs: 학습 에폭 수
        batch_size: 배치 크기
        patience: 조기 종료 인내심
        lr: 학습률
    
    Returns:
        tuple: (학습된 모델, 손실 기록, 예측값)
    """
    X_train, y_train = train_data['X_train'], y_train = train_data['y_train']
    X_test, y_test = test_data['X_test'], y_test = test_data['y_test']
    
    # 데이터로더 준비
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 손실 함수, 최적화기 설정
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    # 학습 로그
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    counter = 0
    
    model = model.to(DEVICE)
    
    print(f"\n=== {model_name} 모델 학습 시작 ===")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            # 순전파
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # 에폭 평균 손실
        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)
        
        # 검증
        model.eval()
        with torch.no_grad():
            X_val = X_test.to(DEVICE)
            y_val = y_test.to(DEVICE)
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs.squeeze(), y_val).item()
            val_losses.append(val_loss)
        
        # 학습률 스케줄러 업데이트
        scheduler.step(val_loss)
        
        # 조기 종료 체크
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"조기 종료 (에폭 {epoch+1}): 검증 손실이 {patience}회 연속 개선되지 않음")
                break
        
        # 진행 상황 출력
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"에폭 {epoch+1}/{epochs}, 훈련 손실: {train_loss:.4f}, 검증 손실: {val_loss:.4f}")
    
    # 최적 모델 복원
    model.load_state_dict(best_model)
    
    # 최종 예측
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test.to(DEVICE)
        predictions = model(X_test_tensor).cpu().numpy().flatten()
    
    # 학습 시간
    train_time = time.time() - start_time
    print(f"{model_name} 모델 학습 완료 (소요시간: {train_time:.2f}초)")
    print(f"최종 검증 손실: {best_val_loss:.4f}")
    
    # 모델 저장
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'{model_name}_model.pth'))
    
    # 손실 곡선 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{model_name}_loss_curve.png'))
    plt.close()
    
    return model, {'train_loss': train_losses, 'val_loss': val_losses}, predictions

def train_arima_model(data, order=(5,1,0), seasonal_order=(0,0,0,0), test_length=None):
    """
    ARIMA 모델 학습 및 예측
    
    Args:
        data: 시계열 데이터 딕셔너리 {'train': 훈련 시계열, 'test': 테스트 시계열}
        order: ARIMA 차수 (p, d, q)
        seasonal_order: 계절성 ARIMA 차수 (P, D, Q, s)
        test_length: 테스트 데이터 길이
    
    Returns:
        tuple: (모델, 예측값)
    """
    print("\n=== ARIMA 모델 학습 시작 ===")
    start_time = time.time()
    
    # 모델 학습
    model = SARIMAX(
        data['train'], 
        order=order, 
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    fitted_model = model.fit(disp=False)
    
    # 모델 요약
    print(f"\nARIMA{order}{seasonal_order} 모델 요약:")
    print(fitted_model.summary().tables[1])
    
    # 예측
    pred = fitted_model.get_forecast(steps=len(data['test']))
    pred_ci = pred.conf_int()
    predictions = pred.predicted_mean
    
    # 학습 시간
    train_time = time.time() - start_time
    print(f"ARIMA 모델 학습 완료 (소요시간: {train_time:.2f}초)")
    
    # 모델 저장
    joblib.dump(fitted_model, os.path.join(OUTPUT_DIR, 'arima_model.pkl'))
    
    return fitted_model, predictions

def train_prophet_model(data, yearly_seasonality=True, weekly_seasonality=True):
    """
    Facebook Prophet 모델 훈련 및 예측
    
    Args:
        data: Prophet용 데이터프레임들 {'train': 훈련 데이터, 'test': 테스트 데이터}
        yearly_seasonality: 연간 계절성 활성화 여부
        weekly_seasonality: 주간 계절성 활성화 여부
    
    Returns:
        tuple: (모델, 예측값)
    """
    if not prophet_available:
        print("Prophet 라이브러리가 설치되어 있지 않아 모델을 학습할 수 없습니다.")
        return None, None
    
    print("\n=== Prophet 모델 학습 시작 ===")
    start_time = time.time()
    
    # Prophet 모델 초기화
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    
    # 추가 리젠서 (주요 한국 휴일)
    model.add_country_holidays(country_name='South Korea')
    
    # 모델 학습
    model.fit(data['train'])
    
    # 예측 기간 설정
    future = model.make_future_dataframe(periods=len(data['test']), freq='D')
    
    # 예측
    forecast = model.predict(future)
    
    # 테스트 기간에 해당하는 예측 추출
    test_dates = data['test']['ds']
    forecast_test = forecast[forecast['ds'].isin(test_dates)]
    predictions = forecast_test['yhat'].values
    
    # 학습 시간
    train_time = time.time() - start_time
    print(f"Prophet 모델 학습 완료 (소요시간: {train_time:.2f}초)")
    
    # 컴포넌트 시각화
    fig = model.plot_components(forecast)
    fig.savefig(os.path.join(OUTPUT_DIR, 'prophet_components.png'))
    plt.close(fig)
    
    # 모델 저장
    with open(os.path.join(OUTPUT_DIR, 'prophet_model.json'), 'w') as f:
        json.dump(model.to_json(), f)
    
    return model, predictions

def train_xgboost_model(X_train, y_train, X_test, params=None):
    """
    XGBoost 모델 학습 및 예측
    
    Args:
        X_train: 훈련 특성
        y_train: 훈련 타겟
        X_test: 테스트 특성
        params: XGBoost 하이퍼파라미터
        
    Returns:
        tuple: (모델, 예측값)
    """
    print("\n=== XGBoost 모델 학습 시작 ===")
    start_time = time.time()
    
    # 기본 하이퍼파라미터
    if params is None:
        params = {
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 5,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'objective': 'reg:squarederror',
            'random_state': RANDOM_SEED
        }
    
    # 모델 초기화 및 학습
    model = xgb.XGBRegressor(**params)
    
    # 조기 종료를 위한 학습
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test[:100], y_train[:100])],  # 메모리 절약을 위해 일부만 사용
        eval_metric='rmse',
        early_stopping_rounds=50,
        verbose=False
    )
    
    # 최적 반복 횟수
    best_iteration = model.best_iteration
    print(f"최적 반복 횟수: {best_iteration}")
    
    # 예측
    predictions = model.predict(X_test)
    
    # 학습 시간
    train_time = time.time() - start_time
    print(f"XGBoost 모델 학습 완료 (소요시간: {train_time:.2f}초)")
    
    # 특성 중요도 시각화
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': [f'Feature_{i}' for i in range(len(feature_importance))],
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
    plt.title('XGBoost 특성 중요도 (상위 20개)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'xgboost_importance.png'))
    plt.close()
    
    # 모델 저장
    model.save_model(os.path.join(OUTPUT_DIR, 'xgboost_model.json'))
    
    return model, predictions

def train_lightgbm_model(X_train, y_train, X_test, y_test, params=None):
    """
    LightGBM 모델 학습 및 예측
    
    Args:
        X_train: 훈련 특성
        y_train: 훈련 타겟
        X_test: 테스트 특성
        y_test: 테스트 타겟
        params: LightGBM 하이퍼파라미터
        
    Returns:
        tuple: (모델, 예측값)
    """
    print("\n=== LightGBM 모델 학습 시작 ===")
    start_time = time.time()
    
    # 기본 하이퍼파라미터
    if params is None:
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'max_depth': -1,
            'bagging_fraction': 0.8,
            'feature_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'random_state': RANDOM_SEED
        }
    
    # 평가 데이터셋
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    # 모델 초기화 및 학습
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    # 최적 반복 횟수
    best_iteration = model.best_iteration_
    print(f"최적 반복 횟수: {best_iteration}")
    
    # 예측
    predictions = model.predict(X_test)
    
    # 학습 시간
    train_time = time.time() - start_time
    print(f"LightGBM 모델 학습 완료 (소요시간: {train_time:.2f}초)")
    
    # 특성 중요도 시각화
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': [f'Feature_{i}' for i in range(len(feature_importance))],
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
    plt.title('LightGBM 특성 중요도 (상위 20개)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'lightgbm_importance.png'))
    plt.close()

# =============================================================================
# 7. 모델 평가 함수
# =============================================================================

def evaluate_predictions(y_true, y_pred, scaler=None, model_name='Model'):
    """
    모델 예측 성능을 평가하는 함수
    
    Args:
        y_true: 실제값
        y_pred: 예측값
        scaler: 스케일러 (사용되었을 경우)
        model_name: 모델 이름
    
    Returns:
        dict: 성능 지표
    """
    # 스케일링 되어 있다면 원래 스케일로 변환
    if scaler is not None:
        y_true_orig = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred
    
    # 성능 지표 계산
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    mape = mean_absolute_percentage_error(y_true_orig, y_pred_orig) * 100  # 퍼센트로 변환
    r2 = r2_score(y_true_orig, y_pred_orig)
    
    # 결과 출력
    print(f"\n{model_name} 성능 지표:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"R^2: {r2:.4f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'y_pred': y_pred_orig,
        'y_true': y_true_orig
    }

# =============================================================================
# 8. 앙상블 모델 구축
# =============================================================================

def create_ensemble_prediction(predictions_dict, weights=None, method='weighted'):
    """
    여러 모델의 예측 결과를 앙상블하여 최종 예측을 생성하는 함수
    
    Args:
        predictions_dict: 모델별 예측값 딕셔너리 {'model_name': predictions, ...}
        weights: 각 모델에 대한 가중치 딕셔너리 (optional)
        method: 앙상블 방법 ('simple', 'weighted', 'stacking')
    
    Returns:
        numpy.ndarray: 앙상블된 예측값
    """
    if method == 'simple':
        # 단순 평균 앙상블
        all_preds = np.array(list(predictions_dict.values()))
        ensemble_pred = np.mean(all_preds, axis=0)
        print("단순 평균 앙상블 생성 완료")
        
    elif method == 'weighted':
        # 가중 평균 앙상블
        if weights is None:
            # 기본값으로 동일한 가중치 사용
            weights = {model: 1/len(predictions_dict) for model in predictions_dict.keys()}
        
        # 가중치 정규화
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # 가중 평균 계산
        ensemble_pred = np.zeros_like(list(predictions_dict.values())[0])
        for model_name, pred in predictions_dict.items():
            ensemble_pred += normalized_weights[model_name] * pred
            
        print("가중 평균 앙상블 생성 완료")
        print("사용된 가중치:", normalized_weights)
    
    elif method == 'stacking':
        # 스태킹 앙상블 (간단한 구현)
        # 스태킹은 일반적으로 다른 모델의 예측을 입력으로 사용하는 메타 모델을 훈련하는 방식
        # 여기서는 단순화를 위해 생략하고 weighted와 동일하게 처리
        print("스태킹 앙상블은 이 구현에서 가중 평균으로 대체됩니다.")
        return create_ensemble_prediction(predictions_dict, weights, 'weighted')
    
    else:
        raise ValueError(f"지원하지 않는 앙상블 방법: {method}")
    
    return ensemble_pred

def optimize_ensemble_weights(predictions_dict, y_true):
    """
    앙상블에 사용할 최적 가중치를 찾는 함수
    
    Args:
        predictions_dict: 모델별 예측값 딕셔너리
        y_true: 실제값
    
    Returns:
        dict: 최적 가중치
    """
    # 간단한 최적화 전략: MAPE의 역수를 가중치로 사용
    weights = {}
    
    for model_name, y_pred in predictions_dict.items():
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        # 성능이 좋을수록(MAPE가 작을수록) 높은 가중치 부여
        weights[model_name] = 1 / (mape + 1e-10)  # 0으로 나누기 방지
    
    # 오류가 매우 큰 모델(이상치)은 가중치를 0으로 설정
    median_weight = np.median(list(weights.values()))
    weights = {k: v if v <= median_weight * 10 else 0 for k, v in weights.items()}
    
    return weights

# =============================================================================
# 9. 결과 시각화
# =============================================================================

def visualize_predictions(dates, y_true, predictions_dict, target_col, output_dir=OUTPUT_DIR):
    """
    여러 모델의 예측 결과를 시각화하는 함수
    
    Args:
        dates: 날짜 배열
        y_true: 실제값
        predictions_dict: 모델별 예측값 딕셔너리
        target_col: 타겟 변수명
        output_dir: 출력 디렉토리
    """
    # 전체 예측 시각화
    plt.figure(figsize=(15, 8))
    
    # 실제값 플롯
    plt.plot(dates, y_true, label='실제값', color='black', linewidth=2)
    
    # 각 모델의 예측값 플롯
    colors = plt.cm.tab10.colors
    for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
        plt.plot(dates, y_pred, label=f'{model_name} 예측', color=colors[i % len(colors)], alpha=0.7)
    
    plt.title(f'{target_col} 예측 결과 비교')
    plt.xlabel('날짜')
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_predictions_comparison.png'))
    plt.close()
    
    # 각 모델별 예측 vs 실제 시각화
    for model_name, y_pred in predictions_dict.items():
        plt.figure(figsize=(12, 6))
        
        plt.plot(dates, y_true, label='실제값', color='black')
        plt.plot(dates, y_pred, label=f'{model_name} 예측', color='blue', alpha=0.7)
        
        # RMSE 계산
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        plt.title(f'{model_name} 예측 vs 실제 (RMSE: {rmse:.4f})')
        plt.xlabel('날짜')
        plt.ylabel(target_col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_prediction_vs_actual.png'))
        plt.close()
    
    print("예측 결과 시각화 완료")