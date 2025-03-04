#!/usr/bin/env python
import argparse
import os
import logging
import time
import json
from main import setup_logging, preprocess_data, apply_engineering_to_df, run_prediction_pipeline
from data_loader import load_data_from_mongo
from utils import clear_cache

def main():
    """
    간편한 실행을 위한 메인 스크립트

    사용법:
      python run.py --clear_cache --use_gpu --output predictions.json
    """
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="유가 예측 시스템 실행")
    parser.add_argument('--look_back', type=int, default=3, help="시계열 윈도우 크기")
    parser.add_argument('--future_steps', type=int, default=7, help="예측할 미래 일 수")
    parser.add_argument('--ensemble_size', type=int, default=3, help="앙상블 모델 수")
    parser.add_argument('--use_gpu', action='store_true', help="GPU 사용 여부")
    parser.add_argument('--clear_cache', action='store_true', help="캐시 삭제 여부")
    parser.add_argument('--output', type=str, default="predictions.json", help="출력 파일명")
    args = parser.parse_args()
    
    # 로깅 설정
    logger = setup_logging()
    logger.info(f"유가 예측 시스템 실행 시작, 인자: {args}")
    
    # 캐시 삭제
    if args.clear_cache:
        clear_cache()
        logger.info("캐시가 삭제되었습니다.")
    
    start_time = time.time()
    
    try:
        # 1. 데이터 로드
        logger.info("MongoDB에서 데이터 로드 중...")
        df = load_data_from_mongo()
        if df.empty:
            logger.error("데이터를 로드할 수 없습니다.")
            return
        logger.info(f"데이터 로드 완료: {df.shape} 레코드")
        
        # 2. 데이터 전처리
        logger.info("데이터 전처리 중...")
        df = preprocess_data(df)
        
        # 3. 타겟 컬럼 설정
        target_cols = ['gasoline', 'premiumGasoline', 'diesel', 'kerosene']
        logger.info(f"타겟 변수: {target_cols}")
        
        # 4. 특성 공학 적용
        logger.info("특성 공학 적용 중...")
        df = apply_engineering_to_df(df, target_cols)
        
        # 5. 예측 수행
        logger.info("예측 수행 중...")
        predictions = run_prediction_pipeline(
            df, 
            target_cols,
            look_back=args.look_back,
            future_steps=args.future_steps,
            ensemble_size=args.ensemble_size,
            use_gpu=args.use_gpu
        )
        
        # 6. 결과 출력
        logger.info("\n===== 예측 결과 =====")
        for region, pred_df in predictions.items():
            logger.info(f"\n지역: {region}")
            for date, row in pred_df.iterrows():
                date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                logger.info(f"{date_str}: " + ", ".join([f"{col}={row[col]:.2f}" for col in pred_df.columns]))
                
        # 7. JSON 파일로 저장
        json_data = {}
        for region, pred_df in predictions.items():
            json_data[region] = {}
            for date, row in pred_df.iterrows():
                date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                json_data[region][date_str] = {col: float(row[col]) for col in pred_df.columns}
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"예측 결과가 {args.output}에 저장되었습니다.")
        
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {str(e)}", exc_info=True)
    
    total_time = time.time() - start_time
    logger.info(f"총 실행 시간: {total_time:.2f}초")

if __name__ == "__main__":
    main()
