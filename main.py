import os
import torch
import logging
import numpy as np  # np를 추가
from datetime import datetime
from ensemble_trainer import run_ensemble_prediction_pipeline
from data_loader import load_data_from_mongo
from model import create_model
from train import train_model

def setup_logging():
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_prediction_pipeline(df, target_cols, model_type='lstm', look_back=3, future_steps=7, ensemble_size=3, use_gpu=True, batch_size=None):
    logger = logging.getLogger(__name__)
    logger.info("예측 파이프라인 시작...")
    
    # 모델 초기화
    input_dim = df.shape[1] - 1  # 타겟 컬럼을 제외한 피처 수
    hidden_dim = 64
    output_dim = len(target_cols)
    
    model = create_model(model_type, input_dim, hidden_dim, output_dim)

    # 데이터 전처리 및 모델 훈련
    X_train, Y_train = df.drop(columns=['date', 'area']), df[target_cols]
    
    # NaN 값 처리: dropna()를 사용하여 NaN 값을 제거한 후, numpy 배열로 변환
    X_train = X_train.dropna().to_numpy()  # NaN 값 제거하고 numpy 배열로 변환
    Y_train = Y_train.dropna().to_numpy()  # NaN 값 제거하고 numpy 배열로 변환
    
    # NumPy 배열을 np.float64로 변환 후 훈련
    model = train_model(model, X_train, Y_train, epochs=50, batch_size=batch_size)

    # 앙상블 예측 및 결과 반환
    predictions = run_ensemble_prediction_pipeline(df, target_cols, look_back, future_steps, ensemble_size, use_gpu, batch_size)
    
    logger.info(f"예측 완료. {len(predictions)}개 지역에 대한 예측 결과 생성됨")
    return predictions

if __name__ == "__main__":
    logger = setup_logging()
    
    # 로깅을 통해 데이터 로드 시작
    logger.info("MongoDB에서 데이터 로드 중...")
    
    try:
        df = load_data_from_mongo()
        if df.empty:
            logger.error("데이터를 로드할 수 없습니다. 데이터가 비어있습니다.")
        else:
            logger.info(f"데이터 로드 완료: {df.shape[0]}개의 레코드")
            target_cols = ['gasoline', 'premiumGasoline', 'diesel', 'kerosene']
            predictions = run_prediction_pipeline(df, target_cols, model_type='lstm')
            logger.info("전체 파이프라인 실행이 완료되었습니다.")
    except Exception as e:
        logger.error(f"데이터 로딩 중 오류 발생: {str(e)}", exc_info=True)
