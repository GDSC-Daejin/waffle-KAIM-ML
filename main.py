import os
import torch
import pandas as pd
import numpy as np
import argparse
import logging
from datetime import datetime
import json
import time
import multiprocessing as mp

from data_loader import load_data_from_mongo
from data_preprocessor import apply_feature_engineering
from ensemble_trainer import OilPriceEnsembleTrainer, run_ensemble_prediction_pipeline
from utils import get_optimal_device_config, cache_result, clear_cache
from monitor import ResourceMonitor

def setup_logging():
    """로깅 설정"""
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

def preprocess_data(df):
    """데이터 전처리"""
    logger = logging.getLogger(__name__)
    logger.info("데이터 전처리 시작...")
    
    # 날짜 처리
    df['date'] = df['date'].str.replace("Date_", "", regex=False)
    df['date'] = pd.to_datetime(df['date'], format="%Y_%m_%d", errors='coerce')
    df = df.sort_values('date')
    
    # 지역 데이터 처리
    if 'area' in df.columns:
        # 리스트 형태의 area를 문자열로 변환
        df['area'] = df['area'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
        # 콤마로 구분된 지역을 별도 행으로 분리
        df['area'] = df['area'].str.split(',')
        df = df.explode('area')
        # 공백 제거
        df['area'] = df['area'].str.strip()
    else:
        df['area'] = 'National'
    
    # 리스트 형태의 값 변환 (평균값 계산)
    for col in df.columns:
        if col not in ['date', 'area']:
            df[col] = df[col].apply(process_list_values)
    
    # 모든 컬럼을 숫자형으로 변환
    for col in df.columns:
        if col not in ['date', 'area']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # NaN 처리
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='any', subset=[col for col in df.columns if col not in ['date', 'area']])
    
    # 중복 제거
    df = df.drop_duplicates(['date', 'area'])
    
    logger.info(f"전처리 완료. 데이터 크기: {df.shape}")
    return df

def process_list_values(x):
    """리스트 형태의 값을 처리"""
    try:
        if isinstance(x, list):
            # 빈 리스트 처리
            if not x:
                return np.nan
                
            # 중첩 리스트 평탄화
            flat_values = []
            
            def flatten(items):
                for item in items:
                    if isinstance(item, list):
                        flatten(item)
                    else:
                        try:
                            flat_values.append(float(item))
                        except (ValueError, TypeError):
                            pass
            
            flatten(x)
            
            # 숫자값이 있으면 평균 반환, 없으면 NaN
            if flat_values:
                return np.mean(flat_values)
            return np.nan
        
        # 숫자로 변환 시도
        return float(x)
    except (ValueError, TypeError):
        return np.nan

def apply_engineering_to_df(df, target_cols):
    """특성 공학 적용"""
    logger = logging.getLogger(__name__)
    logger.info("특성 공학 적용 시작...")
    
    # 전체 데이터에 특성 공학 적용
    enhanced_df = apply_feature_engineering(df, target_cols)
    
    logger.info(f"특성 공학 적용 완료. 특성 수: {len(enhanced_df.columns)}")
    return enhanced_df

@cache_result(expire_hours=24)
def run_prediction_pipeline(df, target_cols, look_back=3, future_steps=7, ensemble_size=3, use_gpu=True, batch_size=None):
    """예측 파이프라인 실행"""
    logger = logging.getLogger(__name__)
    logger.info("예측 파이프라인 시작...")
    
    # 앙상블 예측 실행
    predictions = run_ensemble_prediction_pipeline(
        df,
        target_cols,
        look_back=look_back,
        future_steps=future_steps,
        ensemble_size=ensemble_size,
        use_gpu=use_gpu,
        batch_size=batch_size  # 배치 크기 추가
    )
    
    logger.info(f"예측 완료. {len(predictions)}개 지역에 대한 예측 결과 생성됨")
    return predictions

def export_predictions_to_json(predictions, output_file):
    """예측 결과를 JSON 파일로 저장"""
    logger = logging.getLogger(__name__)
    
    # 데이터프레임을 JSON 직렬화 가능한 형태로 변환
    json_data = {}
    for region, pred_df in predictions.items():
        json_data[region] = {}
        for date, row in pred_df.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            json_data[region][date_str] = row.to_dict()
    
    # JSON 파일 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"예측 결과가 {output_file}에 저장되었습니다.")

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="유가 예측 시스템")
    parser.add_argument('--look_back', type=int, default=int(os.getenv("LOOK_BACK", "3")),
                        help="시계열 윈도우 크기")
    parser.add_argument('--future_steps', type=int, default=int(os.getenv("FUTURE_STEPS", "7")),
                        help="예측할 미래 일 수")
    parser.add_argument('--ensemble_size', type=int, default=int(os.getenv("ENSEMBLE_SIZE", "3")),
                        help="앙상블 모델 수")
    parser.add_argument('--use_gpu', action='store_true', default=(os.getenv("USE_GPU", "true").lower() == "true"),
                        help="GPU 사용 여부")
    parser.add_argument('--clear_cache', action='store_true', help="캐시 삭제 여부")
    parser.add_argument('--output', type=str, default="predictions.json", help="출력 파일명")
    parser.add_argument('--batch_size', type=int, default=None, help="학습 배치 크기 (None: 자동 설정)")
    parser.add_argument('--monitor_resources', action='store_true', help="시스템 리소스 모니터링 활성화")
    args = parser.parse_args()
    
    # 타겟 컬럼 환경 변수에서 가져오기
    target_cols = os.getenv("TARGET_COLS", "gasoline,premiumGasoline,diesel,kerosene").split(",")
    
    # 로깅 설정
    logger = setup_logging()
    logger.info(f"유가 예측 시스템 시작, 인자: {args}")
    
    # 캐시 삭제 요청이 있으면 캐시 삭제
    if args.clear_cache:
        clear_cache()
        logger.info("캐시가 삭제되었습니다.")
    
    # 하드웨어 환경 확인
    device, use_mixed_precision, num_workers = get_optimal_device_config()
    logger.info(f"디바이스: {device}, 혼합 정밀도: {use_mixed_precision}, 워커 수: {num_workers}")
    logger.info(f"CPU 코어 수: {os.cpu_count()}")
    
    start_time = time.time()
    
    # 리소스 모니터링 시작 (선택 사항)
    if args.monitor_resources:
        monitor = ResourceMonitor(interval=1.0)
        monitor.start()
    else:
        monitor = None
    
    # 데이터 로드
    try:
        df = load_data_from_mongo()
        if df.empty:
            logger.error("데이터를 로드할 수 없습니다.")
            return
        logger.info(f"MongoDB에서 데이터 로드 완료. 크기: {df.shape}")
    except Exception as e:
        logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
        return
    
    # 데이터 전처리
    try:
        df = preprocess_data(df)
    except Exception as e:
        logger.error(f"데이터 전처리 중 오류 발생: {str(e)}")
        return
    
    # 특성 공학 적용
    try:
        df = apply_engineering_to_df(df, target_cols)
    except Exception as e:
        logger.error(f"특성 공학 적용 중 오류 발생: {str(e)}")
        return
    
    # 예측 실행
    try:
        predictions = run_prediction_pipeline(
            df,
            target_cols,
            look_back=args.look_back,
            future_steps=args.future_steps,
            ensemble_size=args.ensemble_size,
            use_gpu=args.use_gpu,
            batch_size=args.batch_size  # 배치 크기 추가
        )
    except Exception as e:
        logger.error(f"예측 실행 중 오류 발생: {str(e)}")
        if monitor:
            monitor.stop()
            monitor.plot(save_path="error_resource_usage.png")
        return
    
    # 결과 출력
    logger.info("\n=== 예측 결과 요약 ===")
    for region, pred_df in predictions.items():
        logger.info(f"\n지역: {region}")
        for date, row in pred_df.iterrows():
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            logger.info(f"{date_str}: " + ", ".join([f"{col}={row[col]:.2f}" for col in target_cols]))
    
    # JSON 형식으로 저장
    try:
        export_predictions_to_json(predictions, args.output)
    except Exception as e:
        logger.error(f"결과 저장 중 오류 발생: {str(e)}")
    
    # 리소스 모니터링 종료 및 결과 저장
    if monitor:
        monitor.stop()
        monitor.plot(save_path="resource_usage.png")
        logger.info("리소스 사용량 그래프가 저장되었습니다.")
    
    total_time = time.time() - start_time
    logger.info(f"전체 실행 시간: {total_time:.2f}초")

if __name__ == "__main__":
    main()
