import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import asyncio
import motor.motor_asyncio

def ensure_date_format(df):
    """
    데이터프레임의 date 컬럼이 datetime 형식인지 확인하고, 아니면 변환합니다.
    """
    if 'date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            if isinstance(df['date'].iloc[0], str):
                if df['date'].iloc[0].startswith('Date_'):
                    df['date'] = pd.to_datetime(df['date'].str.replace('Date_', ''), format='%Y_%m_%d')
                else:
                    try:
                        df['date'] = pd.to_datetime(df['date'])
                    except:
                        pass  # 변환 실패하면 그대로 둠
    return df

def merge_with_csv_data(new_data, csv_path, logger=None):
    """
    새로 가져온 데이터를 기존 CSV 데이터와 병합합니다.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    # CSV 파일 로드
    if os.path.exists(csv_path):
        try:
            csv_data = pd.read_csv(csv_path)
            logger.info(f"기존 CSV 파일에서 {len(csv_data)}개 레코드 로드")
            
            # 날짜 필드 처리
            csv_data = ensure_date_format(csv_data)
            new_data = ensure_date_format(new_data)
            
            # 중복 제거를 위해 기존 데이터에서 새 데이터의 날짜 범위에 속하는 레코드를 먼저 제거
            if not new_data.empty and 'date' in new_data.columns and 'date' in csv_data.columns:
                min_date = new_data['date'].min()
                max_date = new_data['date'].max()
                
                # 날짜 범위에 속하는 기존 데이터를 필터링하여 제거
                csv_data = csv_data[
                    (csv_data['date'] < min_date) | 
                    (csv_data['date'] > max_date)
                ]
                
                logger.info(f"날짜 범위 {min_date} ~ {max_date}에 해당하는 기존 레코드 제거")
                
            # 새 데이터와 병합
            merged_df = pd.concat([csv_data, new_data], ignore_index=True)
            
            # 날짜 오름차순 정렬
            merged_df = merged_df.sort_values('date').reset_index(drop=True)
            
            # 중복 제거 (혹시 모를 중복을 대비)
            merged_df = merged_df.drop_duplicates(subset=['date']).reset_index(drop=True)
            
            logger.info(f"데이터 병합 완료: 총 {len(merged_df)}개 레코드")
            
            return merged_df
        except Exception as e:
            logger.error(f"CSV 데이터 병합 중 오류: {str(e)}")
            return new_data if not new_data.empty else None
    else:
        logger.warning(f"기존 CSV 파일이 없습니다: {csv_path}")
        return new_data

def backup_and_save_csv(df, csv_path, cache_dir, logger=None):
    """
    데이터프레임을 CSV로 저장하기 전에 기존 파일을 백업합니다.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    # 기존 파일 백업
    if os.path.exists(csv_path):
        backup_file = os.path.join(
            cache_dir, 
            f"backup_{os.path.basename(csv_path).split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        try:
            import shutil
            shutil.copy2(csv_path, backup_file)
            logger.info(f"기존 파일 백업 완료: {backup_file}")
        except Exception as e:
            logger.error(f"파일 백업 중 오류: {str(e)}")
    
    # 새로운 데이터 저장
    try:
        df.to_csv(csv_path, index=False)
        logger.info(f"데이터를 CSV 파일로 저장 완료: {csv_path}")
        return True
    except Exception as e:
        logger.error(f"CSV 파일 저장 중 오류: {str(e)}")
        return False

def check_data_integrity(df, logger=None):
    """
    데이터 무결성 검사를 수행합니다.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    issues = []
    
    # 날짜 중복 확인
    if df['date'].duplicated().any():
        dup_dates = df[df['date'].duplicated()]['date'].unique()
        issues.append(f"중복된 날짜가 발견되었습니다: {dup_dates}")
        logger.warning(f"중복된 날짜가 {len(dup_dates)}개 발견되었습니다")
    
    # 날짜 간격 체크 (데이터 누락 확인)
    dates = pd.to_datetime(df['date']).sort_values()
    date_diff = dates.diff().dropna()
    missing_days = date_diff[date_diff > pd.Timedelta(days=1)]
    
    if not missing_days.empty:
        for idx in missing_days.index:
            gap = date_diff[idx].days - 1
            if gap > 0:
                from_date = dates[idx-1]
                to_date = dates[idx]
                issues.append(f"데이터 누락: {from_date.strftime('%Y-%m-%d')} 부터 {to_date.strftime('%Y-%m-%d')} 사이 {gap}일")
        
        logger.warning(f"총 {len(missing_days)}개의 날짜 간격에서 데이터가 누락되었습니다")
    
    # NaN 확인
    na_counts = df.isna().sum()
    columns_with_na = na_counts[na_counts > 0]
    if not columns_with_na.empty:
        for col, count in columns_with_na.items():
            issues.append(f"컬럼 '{col}'에 {count}개의 NULL 값이 있습니다 ({count/len(df)*100:.1f}%)")
        
        logger.warning(f"{len(columns_with_na)}개 컬럼에서 NULL 값이 발견되었습니다")
    
    return issues

def get_date_range_summary(df, logger=None):
    """
    데이터의 날짜 범위 요약 정보를 반환합니다.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    if 'date' not in df.columns or df.empty:
        return "날짜 데이터가 없거나 데이터프레임이 비어 있습니다"
    
    # 날짜 정렬
    dates = pd.to_datetime(df['date']).sort_values()
    
    # 날짜 범위 계산
    start_date = dates.min()
    end_date = dates.max()
    date_range = (end_date - start_date).days + 1
    
    # 실제 데이터 일수
    actual_days = len(dates.unique())
    
    # 누락된 일수
    missing_days = date_range - actual_days
    
    # 요약 정보
    summary = f"데이터 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({date_range}일)"
    if missing_days > 0:
        summary += f"\n실제 데이터: {actual_days}일 (누락: {missing_days}일, {missing_days/date_range*100:.1f}%)"
    else:
        summary += f"\n모든 날짜의 데이터가 존재합니다 ({actual_days}일)"
    
    # 데이터 밀도 (월별)
    dates_series = pd.Series(pd.to_datetime(df['date']))
    monthly_counts = dates_series.dt.to_period('M').value_counts().sort_index()
    
    summary += "\n\n월별 데이터 수:"
    for period, count in monthly_counts.items():
        summary += f"\n- {period}: {count}일"
    
    if logger:
        logger.info(summary.split("\n")[0])
    
    return summary

async def async_load_data_by_date_range(db, start_date, end_date):
    """
    날짜 범위를 기준으로 컬렉션에서 데이터를 비동기적으로 검색합니다.
    """
    query = {
        "date": {
            "$gte": start_date.strftime("Date_%Y_%m_%d"),
            "$lte": end_date.strftime("Date_%Y_%m_%d")
        }
    }
    
    all_data = []
    db_collections = await db.list_collection_names()
    date_collections = [c for c in db_collections if c.startswith("Date_")]
    main_logger.info(f"총 {len(date_collections)}개의 날짜 컬렉션 발견")
    
    # 날짜별로 컬렉션 필터링
    target_collections = []
    current_date = start_date
    while current_date <= end_date:
        collection_name = current_date.strftime("Date_%Y_%m_%d")
        if collection_name in date_collections:
            target_collections.append(collection_name)
        current_date += timedelta(days=1)
    
    main_logger.info(f"조회 기간 내 {len(target_collections)}개의 유효한 컬렉션 확인")
    
    # 배치 크기 정의 (한 번에 처리할 컬렉션 수)
    batch_size = 30
    batches = [target_collections[i:i+batch_size] for i in range(0, len(target_collections), batch_size)]
    
    total_docs = 0
    for batch_idx, collection_batch in enumerate(batches):
        main_logger.info(f"배치 {batch_idx+1}/{len(batches)} 처리 중 ({len(collection_batch)}개 컬렉션)")
        
        # 이 배치의 모든 컬렉션에 대한 조회 작업 생성
        tasks = []
        for coll_name in collection_batch:
            task = asyncio.create_task(async_fetch_collection(db, coll_name))
            tasks.append(task)
        
        # 모든 작업이 완료될 때까지 대기
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 처리
        for coll_name, result in zip(collection_batch, batch_results):
            if isinstance(result, Exception):
                main_logger.error(f"컬렉션 {coll_name} 조회 실패: {result}")
                continue
                
            if result:  # 결과가 비어있지 않으면
                all_data.extend(result)
                total_docs += len(result)
        
        main_logger.info(f"배치 {batch_idx+1} 완료: 누적 {total_docs}개 문서")

    main_logger.info(f"전체 조회 완료: {total_docs}개 문서")
    return all_data

async def async_fetch_collection(db, collection_name):
    """
    지정된 컬렉션에서 모든 문서를 비동기적으로 가져옵니다.
    """
    try:
        cursor = db[collection_name].find({}, {"_id": 0})
        documents = await cursor.to_list(length=1000)  # 최대 1000개 문서 가져오기 (필요시 조정)
        main_logger.info(f"컬렉션 {collection_name}: {len(documents)}개 문서 조회됨")
        return documents
    except Exception as e:
        main_logger.error(f"컬렉션 {collection_name} 조회 중 오류: {str(e)}")
        raise

@memory.cache
def load_data_from_mongo_new(mongo_uri, db_name, start_date, end_date):
    """
    비동기 방식으로 MongoDB에서 데이터를 로드합니다.
    """
    main_logger.info("MongoDB 데이터 로드 시작 (비동기 로드 + 배치 처리)")
    
    try:
        # 비동기 클라이언트 생성
        client = motor.motor_asyncio.AsyncIOMotorClient(
            mongo_uri,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=20000,
            socketTimeoutMS=35000,
            maxPoolSize=50,
            minPoolSize=10
        )
        db = client[db_name]
        
        # 비동기 실행
        loop = asyncio.get_event_loop()
        documents = loop.run_until_complete(async_load_data_by_date_range(db, start_date, end_date))
        
        # 결과 처리
        if documents:
            df = pd.DataFrame(documents)
            main_logger.info(f"MongoDB 데이터 로드 완료: {df.shape}")
            return df
        else:
            main_logger.error("데이터를 찾을 수 없습니다.")
            return pd.DataFrame()
    except Exception as e:
        main_logger.error(f"MongoDB 데이터 로드 중 오류 발생: {str(e)}")
        return pd.DataFrame()
    finally:
        try:
            client.close()
        except:
            pass
