import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import matplotlib.pyplot as plt
import shap
from tqdm import tqdm

def normalize_data(data):
    """
    [원리 설명]
    - 입력 데이터의 각 컬럼을 0과 1 사이로 정규화합니다.
    - MinMaxScaler를 사용해 각 열의 최소/최대값을 기준으로 선형 변환을 수행합니다.
    
    파라미터:
      - data: 정규화할 numpy 배열
    반환:
      - 정규화된 데이터와 이후 역정규화에 사용할 scaler 객체
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data), scaler

def create_dataset(dataset, look_back=1):
    """
    [원리 설명]
    - 시계열 데이터를 학습용 데이터셋으로 변환하는 함수입니다.
    - 연속된 look_back 기간의 데이터를 하나의 입력(X) 샘플로 구성하고,
      그 바로 다음 시점의 데이터를 타겟(Y)으로 설정합니다.
      
    예:
      - look_back=3이면, 인덱스 0~2를 X, 3번째 데이터를 Y로 사용합니다.
      - 이후 1~3을 X, 4번째를 Y로 하는 식으로 슬라이딩 윈도우를 구성합니다.
    
    파라미터:
      - dataset: 정규화된 전체 데이터 (numpy 배열)
      - look_back: 입력 시퀀스 길이 (일수)
    반환:
      - X: (샘플수, look_back, 특성 수) 배열
      - Y: (샘플수, 특성 수) 배열
    """
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back])
        Y.append(dataset[i+look_back])
    return np.array(X), np.array(Y)

def prepare_data(X, Y):
    """
    [원리 설명]
    - numpy 배열 형태의 데이터를 PyTorch 텐서로 변환합니다.
    - LSTM 모델 학습에 사용하기 위해 torch.FloatTensor 형태로 바꿉니다.
    
    파라미터:
      - X: 입력 데이터 (numpy 배열)
      - Y: 타겟 데이터 (numpy 배열)
    반환:
      - (X, Y) 형태의 PyTorch 텐서
    """
    return torch.FloatTensor(X), torch.FloatTensor(Y)

def split_train_test(X, Y, test_size=0.2):
    """
    시계열 데이터를 시간 순서대로 분할합니다.
    
    파라미터:
      - X: 입력 데이터 (numpy 배열)
      - Y: 타겟 데이터 (numpy 배열)
      - test_size: 테스트 데이터 비율 (기본값 0.2)
    반환:
      - X_train: 학습용 입력 데이터
      - Y_train: 학습용 타겟 데이터
      - X_test: 테스트용 입력 데이터
      - Y_test: 테스트용 타겟 데이터
    """
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    return X_train, Y_train, X_test, Y_test

def analyze_feature_importance(model, X, feature_names, target_names, n_samples=100):
    """
    SHAP 값을 사용하여 모델의 특성 중요도를 분석합니다.
    
    파라미터:
      - model: 학습된 PyTorch 모델
      - X: 특성 데이터
      - feature_names: 특성 이름 리스트
      - target_names: 타겟 이름 리스트
      - n_samples: SHAP 값 계산에 사용할 샘플 수 (기본값 100)
    반환:
      - 특성 중요도 데이터프레임
    """
    model.eval()
    
    # 배경 데이터 (샘플링)
    background_indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    background = X[background_indices]
    
    # PyTorch 모델을 래핑하는 함수
    def model_predict(x):
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            return model(x_tensor).numpy()
    
    # SHAP 설명자 생성
    explainer = shap.DeepExplainer(model, torch.FloatTensor(background))
    
    # 모든 데이터에 대한 SHAP 값 계산
    sample_indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    sample_data = X[sample_indices]
    shap_values = explainer.shap_values(torch.FloatTensor(sample_data))
    
    # 시계열 데이터 중 마지막 시점만 사용
    feature_importance = np.zeros((len(feature_names), len(target_names)))
    
    for target_idx in range(len(target_names)):
        target_shap = shap_values[target_idx]
        
        # 각 특성별로 중요도 계산 (절대값의 평균)
        for feature_idx in range(len(feature_names)):
            # 시계열의 모든 시점에 대해 평균
            feature_importance[feature_idx, target_idx] = np.abs(target_shap[:, :, feature_idx]).mean()
    
    # 결과를 데이터프레임으로 변환
    importance_df = pd.DataFrame(feature_importance, index=feature_names, columns=target_names)
    return importance_df

def plot_feature_importance(importance_df, top_n=15, figsize=(12, 10)):
    """
    특성 중요도를 시각화합니다.
    
    파라미터:
      - importance_df: analyze_feature_importance 함수의 반환값
      - top_n: 표시할 상위 특성 수
      - figsize: 그림 크기
    """
    targets = importance_df.columns
    n_targets = len(targets)
    
    plt.figure(figsize=figsize)
    
    for i, target in enumerate(targets):
        plt.subplot(n_targets, 1, i+1)
        
        # 상위 N개 특성 선택
        sorted_idx = importance_df[target].argsort()
        top_features = importance_df.iloc[sorted_idx[-top_n:]]
        
        # 수평 막대 그래프
        ax = top_features[target].plot(kind='barh')
        ax.set_title(f"Top {top_n} Features for {target}", fontsize=12)
        ax.set_xlabel('Feature Importance (SHAP value)')
        ax.set_ylabel('Features')
        plt.tight_layout()
    
    plt.tight_layout()
    plt.show()

def select_optimal_features(X, Y, feature_names, target_names, test_size=0.2, cv_folds=5, 
                           importance_threshold=0.1, verbose=True):
    """
    최적의 특성을 선택합니다.
    
    파라미터:
      - X, Y: 입력 데이터와 타겟 데이터
      - feature_names: 특성 이름 리스트
      - target_names: 타겟 이름 리스트
      - test_size: 테스트 세트 비율
      - cv_folds: 교차 검증 폴드 수
      - importance_threshold: 중요도 임계값 (이 값보다 낮은 특성은 제거)
      - verbose: 상세 출력 여부
    
    반환:
      - 선택된 특성의 인덱스
    """
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import TimeSeriesSplit
    
    # 시계열 데이터 분할
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # 선택된 특성 인덱스
    selected_features = []
    
    # 각 타겟에 대해 특성 선택
    for i in range(Y.shape[1]):
        y = Y[:, i]
        
        # 랜덤 포레스트를 사용한 점진적 특성 선택
        sfs = SequentialFeatureSelector(
            RandomForestRegressor(n_estimators=100, random_state=42),
            n_features_to_select='auto',
            direction='forward',
            scoring='neg_mean_squared_error',
            cv=tscv
        )
        
        # 특성 선택 수행
        sfs.fit(X.reshape(X.shape[0], -1), y)
        
        # 선택된 특성 인덱스 저장
        selected = np.where(sfs.get_support())[0]
        selected_features.append(selected)
        
        if verbose:
            print(f"Target {target_names[i]}: Selected {len(selected)} features")
            print(f"Selected features: {[feature_names[j] for j in selected]}")
    
    # 모든 타겟에 대해 선택된 특성 합집합
    combined_selected = np.unique(np.concatenate(selected_features))
    
    if verbose:
        print(f"Total selected features: {len(combined_selected)}/{len(feature_names)}")
    
    return combined_selected

def feature_selection_pipeline(X, Y, feature_names, target_names, look_back=3, test_size=0.2):
    """
    특성 선택 파이프라인 함수
    
    파라미터:
      - X, Y: 시계열 데이터 (3D 및 2D)
      - feature_names: 특성 이름 리스트
      - target_names: 타겟 이름 리스트
      - look_back: 시퀀스 길이
      - test_size: 테스트 세트 비율
    
    반환:
      - 선택된 특성을 반영한 X 데이터
      - 선택된 특성 이름 리스트
    """
    X_train, _, X_test, _ = split_train_test(X, Y, test_size)
    
    # LSTM 모델을 학습하여 특성 중요도 분석
    from model import LSTMModel
    from train import train_model
    
    input_dim = X_train.shape[2]
    hidden_dim = 50
    layer_dim = 1
    output_dim = Y.shape[1]
    
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    model, _ = train_model(model, X_train, torch.FloatTensor(Y[:len(X_train)]), epochs=50, verbose=0)
    
    # SHAP으로 특성 중요도 분석
    importance_df = analyze_feature_importance(model, X_test.numpy(), feature_names, target_names)
    
    # 중요도가 높은 특성만 선택
    threshold = importance_df.mean(axis=1).median()
    selected_features = importance_df[importance_df.mean(axis=1) > threshold].index.tolist()
    
    print(f"Selected {len(selected_features)}/{len(feature_names)} features based on importance")
    
    # 선택된 특성만 포함하는 새 X 데이터 생성
    selected_indices = [feature_names.index(f) for f in selected_features]
    X_selected = X[:, :, selected_indices]
    
    return X_selected, selected_features

def generate_lag_features(data, lag_cols, lags=[1, 2, 3]):
    """
    시계열 지연 특성을 생성합니다.
    
    파라미터:
      - data: 원본 데이터프레임
      - lag_cols: 지연 특성을 생성할 컬럼 리스트
      - lags: 지연 기간 리스트
    
    반환:
      - 지연 특성이 추가된 데이터프레임
    """
    df = data.copy()
    for col in lag_cols:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # NaN 값 제거
    df = df.dropna()
    return df

def generate_rolling_features(data, roll_cols, windows=[3, 7, 14]):
    """
    이동 평균/표준편차 특성을 생성합니다.
    
    파라미터:
      - data: 원본 데이터프레임
      - roll_cols: 이동 특성을 생성할 컬럼 리스트
      - windows: 윈도우 크기 리스트
    
    반환:
      - 이동 특성이 추가된 데이터프레임
    """
    df = data.copy()
    
    for col in roll_cols:
        for window in windows:
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window).mean()
            df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window).std()
    
    # NaN 값 제거
    df = df.dropna()
    return df

def apply_feature_engineering(df, target_cols):
    """특성 공학 파이프라인 함수"""
    # 원본 데이터 복사
    df = df.reset_index(drop=True)
    
    # 특성 저장을 위한 딕셔너리 생성 (프래그먼테이션 방지)
    feature_dict = {}
    
    # 날짜 관련 특성 추가
    if 'date' in df.columns:
        feature_dict['day_of_week'] = df['date'].dt.dayofweek.values
        feature_dict['month'] = df['date'].dt.month.values
        feature_dict['quarter'] = df['date'].dt.quarter.values
        feature_dict['year'] = df['date'].dt.year.values
        feature_dict['day_of_year'] = df['date'].dt.dayofyear.values
    
    # 국제 유가 관련 컬럼 식별
    oil_cols = [col for col in df.columns if any(s in col.lower() for s in ['dubai', 'brent', 'wti'])]
    
    # 경제 지표 관련 컬럼 식별
    econ_cols = [col for col in df.columns if any(s in col.lower() for s in ['gdp', 'cpi', 'interest', 'krw'])]
    
    # 지연 특성 생성
    lag_features = oil_cols + econ_cols + target_cols
    for col in lag_features:
        if col in df.columns:
            for lag in [1, 2, 3, 7]:
                feature_dict[f'{col}_lag_{lag}'] = df[col].shift(lag).values
    
    # 이동 평균/표준편차 특성 생성
    rolling_features = {}
    for col in lag_features:
        if col in df.columns:
            for window in [3, 7, 14]:
                rolling_mean = df[col].rolling(window=window).mean().values
                rolling_std = df[col].rolling(window=window).std().values
                feature_dict[f'{col}_roll_mean_{window}'] = rolling_mean
                feature_dict[f'{col}_roll_std_{window}'] = rolling_std
    
    # 유가 비율 특성 (국제 유가간 비율)
    ratio_features = {}
    if len(oil_cols) >= 2:
        for i, col1 in enumerate(oil_cols):
            for col2 in oil_cols[i+1:]:
                feature_dict[f'{col1}_to_{col2}_ratio'] = (df[col1] / df[col2].replace(0, np.nan)).values
    
    # 국제 유가와 국내 유가 간의 비율/차이
    for oil_col in oil_cols:
        for target_col in target_cols:
            if oil_col in df.columns and target_col in df.columns:
                feature_dict[f'{oil_col}_to_{target_col}_ratio'] = (df[oil_col] / df[target_col].replace(0, np.nan)).values
                feature_dict[f'{oil_col}_to_{target_col}_diff'] = (df[oil_col] - df[target_col]).values
    
    # 모든 새로운 특성을 한 번에 DataFrame으로 변환 (프래그먼테이션 방지)
    new_features_df = pd.DataFrame(feature_dict, index=df.index)
    
    # 원본 데이터프레임과 효율적으로 병합
    new_df = pd.concat([df, new_features_df], axis=1)
    
    # NaN 값 효율적 처리 (권장 메서드 사용)
    new_df = new_df.ffill().bfill()
    
    return new_df