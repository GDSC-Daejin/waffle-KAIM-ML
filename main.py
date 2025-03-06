import os
import sys
import time
import torch
import logging
import numpy as np
import pandas as pd
import psutil
import json  # json 모듈 추가
from datetime import datetime
from ensemble_trainer import run_ensemble_prediction_pipeline
from data_loader import load_data_from_mongo
from data_preprocessor import preprocess_data
from logging_utils import CustomProgressBar, PrettyLogger
from parallel_utils import accelerate_training
from utils import optimize_memory_usage
from dotenv import load_dotenv  # dotenv 추가

# .env 파일 로드
load_dotenv()

# 환경 변수에서 설정 읽기
EPOCHS = int(os.getenv("EPOCHS", "100"))
IMPORTANCE_EPOCHS = int(os.getenv("IMPORTANCE_EPOCHS", "50"))
ENSEMBLE_SIZE = int(os.getenv("ENSEMBLE_SIZE", "3"))
LOOK_BACK = int(os.getenv("LOOK_BACK", "3"))
FUTURE_STEPS = int(os.getenv("FUTURE_STEPS", "7"))

def setup_logging():
    """로깅 환경 설정"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 현재 시간을 포함한 로그 파일명 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"main_{timestamp}.log")
    
    # 로깅 설정
    logger = PrettyLogger("KAIM-ML", log_dir=log_dir).logger
    
    # 시스템 정보 로깅
    logger.info("\n" + "="*50)
    logger.info("유가 예측 시스템 시작")
    logger.info("="*50)
    
    # CPU 정보
    cpu_cores = os.cpu_count()
    cpu_info = psutil.cpu_freq()
    if cpu_info:
        logger.info(f"CPU: {cpu_cores}코어, 최대 주파수 {cpu_info.max:.2f}MHz")
    else:
        logger.info(f"CPU: {cpu_cores}코어")
    
    # RAM 정보
    ram = psutil.virtual_memory()
    logger.info(f"메모리: 총 {ram.total/(1024**3):.1f}GB, 사용 가능 {ram.available/(1024**3):.1f}GB")
    
    # GPU 정보
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"GPU: {gpu_count}개 발견")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            logger.info(f"  - GPU {i}: {gpu_name}")
    else:
        logger.info("GPU: 사용 불가")
    
    return logger

def run_prediction_pipeline(df, target_cols, look_back=None, future_steps=None, ensemble_size=None, use_gpu=True, batch_size=None):
    """
    유가 예측 파이프라인을 실행합니다.
    
    Args:
        df: 입력 데이터프레임
        target_cols: 예측할 타겟 컬럼 리스트
        look_back: 시계열 윈도우 크기
        future_steps: 예측할 미래 일수
        ensemble_size: 앙상블에 사용할 모델 수
        use_gpu: GPU 사용 여부
        batch_size: 학습 배치 크기
        
    Returns:
        지역별 예측 결과
    """
    # 환경 변수 기본 값 사용
    if look_back is None:
        look_back = LOOK_BACK
    if future_steps is None:
        future_steps = FUTURE_STEPS
    if ensemble_size is None:
        ensemble_size = ENSEMBLE_SIZE

    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*50)
    logger.info("예측 파이프라인 시작")
    logger.info("="*50)
    
    # 학습 가속화 설정 적용
    accelerate_training()
    
    # 1. 데이터 전처리
    logger.info("1단계: 데이터 전처리")
    
    # 리스트 형태의 컬럼 확인
    list_cols = []
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            list_cols.append(col)
    
    if list_cols:
        logger.info(f"리스트 형태의 값이 있는 컬럼: {', '.join(list_cols)}")
    
    # 데이터 전처리
    df = preprocess_data(df)
    logger.info(f"전처리 완료: {df.shape[0]}행 x {df.shape[1]}열")
    
    # 2. 배치 크기 최적화
    if use_gpu and torch.cuda.is_available() and batch_size is None:
        from parallel_utils import optimize_gpu_tensor_ops
        batch_size = optimize_gpu_tensor_ops(mixed_precision=True)
        logger.info(f"GPU 최적화 배치 크기: {batch_size}")
    elif batch_size is None:
        # 대용량 RAM 활용 최적화 (128GB RAM 고려)
        total_ram = psutil.virtual_memory().total / (1024**3)  # GB
        batch_size = min(2048, int(total_ram / 16))  # RAM 크기에 맞춤
        logger.info(f"RAM 최적화 배치 크기: {batch_size}")
    
    # 3. 앙상블 예측 파이프라인 실행
    logger.info("2단계: 앙상블 모델 학습 및 예측")
    try:
        predictions = run_ensemble_prediction_pipeline(
            features_df=df,
            target_cols=target_cols,
            look_back=look_back,
            future_steps=future_steps,
            ensemble_size=ensemble_size,
            use_gpu=use_gpu,
            batch_size=batch_size
        )
        logger.info(f"예측 완료: {len(predictions)}개 지역의 향후 {future_steps}일 예측")
    except Exception as e:
        logger.error(f"예측 과정 중 오류 발생: {str(e)}", exc_info=True)
        raise
    
    # 4. 결과 저장 및 시각화
    logger.info("3단계: 예측 결과 저장")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # CSV 및 JSON 형식으로 결과 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # CSV 저장
    for region, pred_df in predictions.items():
        csv_file = os.path.join(results_dir, f'{region}_predictions_{timestamp}.csv')
        pred_df.to_csv(csv_file)
        logger.info(f"{region} 지역 예측 결과가 {csv_file}에 저장되었습니다.")
    
    # JSON 형식 저장
    result_dict = {}
    for region, pred_df in predictions.items():
        # DataFrame을 딕셔너리로 변환
        result_dict[region] = {}
        for col in pred_df.columns:
            result_dict[region][col] = {}
            for date, value in zip(pred_df.index.strftime('%Y-%m-%d'), pred_df[col].values):
                result_dict[region][col][date] = float(value)
    
    json_file = os.path.join(results_dir, f'predictions_{timestamp}.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    
    logger.info(f"통합 예측 결과가 {json_file}에 저장되었습니다.")
    
    return predictions

if __name__ == "__main__":
    # 메모리 최적화 수행
    optimize_memory_usage()
    
    # 로깅 설정
    logger = setup_logging()
    
    # 진행상황 표시를 위한 단계 정의
    pipeline_steps = [
        "데이터 로드",
        "데이터 전처리",
        "모델 학습",
        "예측 수행",
        "결과 저장"
    ]
    
    # 진행 상태 표시 객체 생성
    progress_bar = CustomProgressBar(len(pipeline_steps), "유가 예측 파이프라인")
    
    try:
        # 1단계: MongoDB에서 데이터 로드
        logger.info("\n1단계: MongoDB에서 데이터 로드")
        df = load_data_from_mongo()
        
        if df.empty:
            logger.error("데이터를 로드할 수 없습니다. 데이터가 비어있습니다.")
            sys.exit(1)
        
        # 데이터 기본 정보 로깅
        logger.info(f"데이터 로드 완료: {df.shape[0]}개의 레코드, {df.shape[1]}개의 컬럼")
        
        if 'date' in df.columns:
            min_date = df['date'].min()
            max_date = df['date'].max()
            logger.info(f"데이터 기간: {min_date} ~ {max_date}")
        
        progress_bar.update(1)
        
        # 타겟 컬럼 설정
        target_cols = ['gasoline', 'premiumGasoline', 'diesel', 'kerosene']
        
        # 타겟 컬럼이 모두 있는지 확인
        missing_cols = [col for col in target_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"다음 타겟 컬럼이 데이터에 없습니다: {missing_cols}")
            sys.exit(1)
        
        # GPU 사용 여부 확인
        use_gpu = torch.cuda.is_available()
        
        # 시작 시간 기록
        start_time = time.time()
        
        # 예측 파이프라인 실행
        predictions = run_prediction_pipeline(
            df=df, 
            target_cols=target_cols, 
            look_back=3, 
            future_steps=7, 
            ensemble_size=3,
            use_gpu=use_gpu
        )
        
        # 실행 시간 계산 및 출력
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info("\n" + "="*50)
        logger.info("유가 예측 작업 완료")
        logger.info(f"총 소요 시간: {int(hours)}시간 {int(minutes)}분 {seconds:.1f}초")
        logger.info("="*50)
        
        # 진행률 100%로 설정
        progress_bar.update(len(pipeline_steps) - progress_bar.current)
        
    except Exception as e:
        logger.error(f"예측 작업 중 오류 발생: {str(e)}", exc_info=True)
        
        # 오류 발생 시 진행 표시줄에 오류 표시
        sys.stdout.write(f"\r파이프라인 진행: |{'!'*50}| 오류 발생\n")
        sys.exit(1)
