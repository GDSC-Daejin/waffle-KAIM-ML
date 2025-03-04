import torch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os
import concurrent.futures

def make_predictions(model, X):
    """
    [원리 설명]
    - 학습된 모델을 사용하여 입력 데이터 X에 대한 예측값을 생성합니다.
    - torch.no_grad()를 사용하여 예측 시 그래디언트 계산을 생략함으로써 효율성을 높입니다.
    
    반환:
      - 예측 결과를 numpy 배열로 반환합니다.
    """
    model.eval()
    with torch.no_grad():
        return model(X).numpy()

def calculate_rmse(y_true, y_pred):
    """
    [원리 설명]
    - 실제 값(y_true)과 예측 값(y_pred) 사이의 평균제곱근오차(RMSE)를 계산합니다.
    - RMSE는 예측 오차의 크기를 나타내며, 값이 낮을수록 예측 성능이 우수함을 의미합니다.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def predict_future(model, last_sequence, future_steps, scaler):
    """
    [원리 설명]
    - 마지막 look_back 기간의 정규화된 데이터를 기반으로 미래 future_steps일치 예측값을 생성합니다.
    - 각 예측 후 슬라이딩 윈도우 방식을 적용하여, 입력 시퀀스를 갱신하고 연속적인 미래 값을 예측합니다.
    - 마지막에 scaler.inverse_transform()을 사용해, 정규화된 예측값을 원래 스케일로 복원합니다.
    
    파라미터:
      - model: 학습된 모델
      - last_sequence: 마지막 look_back일의 정규화된 데이터 (shape: (look_back, num_features))
      - future_steps: 예측할 미래 일 수
      - scaler: MinMaxScaler 객체 (역정규화용)
      
    반환:
      - 미래 예측 결과 (원래 스케일, numpy 배열, shape: (future_steps, num_features))
    """
    model.eval()
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(future_steps):
        input_tensor = torch.FloatTensor(current_seq).unsqueeze(0)
        with torch.no_grad():
            pred = model(input_tensor).numpy()[0]
        predictions.append(pred)
        
        # 슬라이딩: 현재 시퀀스의 첫 행 제거, 새 예측값 추가
        current_seq = np.vstack([current_seq[1:], pred])
    
    predictions = np.array(predictions)
    
    # 예측값이 전체 특성의 일부인 경우 (타겟 컬럼만 예측하는 경우)
    if predictions.shape[1] != current_seq.shape[1]:
        # 전체 특성 크기에 맞는 빈 배열 생성
        full_predictions = np.zeros((future_steps, current_seq.shape[1]))
        target_indices = list(range(predictions.shape[1]))  # 단순화를 위해 0부터 시작하는 인덱스 사용
        
        # 타겟 인덱스 위치에 예측값 삽입
        for i, pred in enumerate(predictions):
            full_predictions[i, target_indices] = pred
        
        # 역정규화
        full_predictions = scaler.inverse_transform(full_predictions)
        
        # 타겟 컬럼만 추출하여 반환
        return full_predictions[:, target_indices]
    else:
        # 전체 특성을 예측한 경우
        return scaler.inverse_transform(predictions)

def ensemble_predict_future(ensemble_model, last_sequence, future_steps, scaler, target_indices=None):
    """
    앙상블 모델을 사용하여 미래 예측 수행
    
    파라미터:
      - ensemble_model: EnsembleModel 인스턴스
      - last_sequence: 마지막 look_back일의 정규화된 데이터
      - future_steps: 예측할 미래 일 수
      - scaler: MinMaxScaler 객체 (역정규화용)
      - target_indices: 타겟 변수의 인덱스 리스트
      
    반환:
      - 미래 예측 결과 (원래 스케일)
    """
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(future_steps):
        # 현재 시퀀스에서 예측
        input_tensor = torch.FloatTensor(current_seq).unsqueeze(0)
        pred = ensemble_model.predict(input_tensor)[0]
        predictions.append(pred)
        
        # 시퀀스 업데이트: 마지막 항목 제거하고 예측값 추가
        if target_indices:
            new_row = np.zeros((1, current_seq.shape[1]))
            for i, target_idx in enumerate(target_indices):
                new_row[0, target_idx] = pred[i]
            current_seq = np.vstack([current_seq[1:], new_row])
        else:
            # 타겟 인덱스가 없으면 예측값을 그대로 사용
            current_seq = np.vstack([current_seq[1:], pred.reshape(1, -1)])
    
    predictions = np.array(predictions)
    
    # 예측값이 전체 특성의 일부인 경우
    if target_indices:
        # 전체 특성 크기에 맞는 빈 배열 생성
        full_predictions = np.zeros((future_steps, current_seq.shape[1]))
        
        # 타겟 인덱스 위치에 예측값 삽입
        for i in range(future_steps):
            for j, idx in enumerate(target_indices):
                full_predictions[i, idx] = predictions[i, j]
        
        # 역정규화
        full_predictions = scaler.inverse_transform(full_predictions)
        
        # 타겟 컬럼만 추출하여 반환
        target_predictions = np.zeros((future_steps, len(target_indices)))
        for i in range(future_steps):
            for j, idx in enumerate(target_indices):
                target_predictions[i, j] = full_predictions[i, idx]
        return target_predictions
    else:
        # 전체 특성을 예측한 경우
        return scaler.inverse_transform(predictions)

def parallel_region_predictions(ensemble_models, region_data, look_back, future_steps, target_cols, n_jobs=-1):
    """
    여러 지역에 대한 예측을 병렬로 수행
    
    파라미터:
      - ensemble_models: 지역별 앙상블 모델 딕셔너리
      - region_data: 지역별 데이터 딕셔너리
      - look_back: 시계열 윈도우 크기
      - future_steps: 예측할 미래 일 수
      - target_cols: 타겟 컬럼 리스트
      - n_jobs: 병렬 작업 수
    
    반환:
      - 지역별 예측 결과 딕셔너리
    """
    def predict_for_region(region):
        # 지역별 모델과 데이터 가져오기
        model = ensemble_models.get(region)
        data = region_data.get(region)
        
        if model is None or data is None:
            print(f"Warning: Model or data not found for region {region}")
            return region, None
        
        # 정규화 및 시계열 데이터셋 생성
        from data_preprocessor import normalize_data, create_dataset, prepare_data
        
        # 'date', 'area' 컬럼 제거
        if 'date' in data.columns:
            data = data.drop(columns=['date'])
        if 'area' in data.columns:
            data = data.drop(columns=['area'])
        
        # 데이터 정규화
        data_array = data.values.astype(float)
        normalized_data, scaler = normalize_data(data_array)
        
        # 마지막 시퀀스
        last_sequence = normalized_data[-look_back:]
        
        # 타겟 인덱스 계산
        col_names = data.columns.tolist()
        target_indices = [col_names.index(col) for col in target_cols if col in col_names]
        
        # 예측 수행
        predictions = ensemble_predict_future(
            model, last_sequence, future_steps, scaler, target_indices
        )
        
        # 데이터프레임 생성
        from datetime import datetime, timedelta
        last_date = datetime.now()  # 기본값
        if 'date' in region_data.get(region).columns:
            last_date = region_data.get(region)['date'].max()
        
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_steps)
        pred_df = pd.DataFrame(predictions, columns=[col for col in target_cols if col in data.columns], index=future_dates)
        
        return region, pred_df
    
    # 병렬 처리 설정
    n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
    regions = list(ensemble_models.keys())
    
    # 병렬 처리 실행
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(predict_for_region, regions))
    
    # 결과 딕셔너리 생성
    predictions = {}
    for region, pred in results:
        if pred is not None:
            predictions[region] = pred
    
    return predictions

def evaluate_ensemble(ensemble_model, X_test, Y_test, target_names=None):
    """
    앙상블 모델의 성능 평가
    
    파라미터:
      - ensemble_model: 앙상블 모델
      - X_test: 테스트 입력 데이터
      - Y_test: 테스트 타겟 데이터
      - target_names: 타겟 변수 이름 리스트
    
    반환:
      - 성능 지표 딕셔너리
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # 예측 수행
    X_test_tensor = torch.FloatTensor(X_test)
    preds = ensemble_model.predict(X_test_tensor)
    
    # 성능 지표 계산
    metrics = {}
    for i, target in enumerate(target_names or [f'target_{i}' for i in range(Y_test.shape[1])]):
        y_true = Y_test[:, i].numpy() if isinstance(Y_test, torch.Tensor) else Y_test[:, i]
        y_pred = preds[:, i]
        
        metrics[target] = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        }
    
    return metrics
