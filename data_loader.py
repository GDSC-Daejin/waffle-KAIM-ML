import pandas as pd
import re
from pymongo import MongoClient
import datetime
import os
import logging
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def load_data_from_mongo():
    """
    [원리 설명]
    - MongoDB 서버에 저장된 'kaim' 데이터베이스에서 데이터를 가져옵니다.
    - 서버 타임아웃을 방지하기 위해 list_collection_names() 대신 
      날짜 범위를 기반으로 컬렉션 이름을 직접 생성합니다.
    - 각 날짜별로 저장된 컬렉션의 이름은 "Date_YYYY_MM_DD" 형식을 따릅니다.
    - 이 함수는 해당 형식의 컬렉션들을 찾아 각 컬렉션의 데이터를 DataFrame으로 변환한 후,
      이를 하나의 DataFrame으로 합쳐 반환합니다.
    """
    logger = logging.getLogger(__name__)
    
    # 환경 변수에서 DB 연결 정보 가져오기
    mongo_user = os.getenv("MONGO_USER")
    mongo_password = os.getenv("MONGO_PASSWORD")
    mongo_host = os.getenv("MONGO_HOST")
    mongo_db = os.getenv("MONGO_DB", "kaim")
    mongo_connect_timeout = int(os.getenv("MONGO_CONNECT_TIMEOUT", "120000"))
    mongo_socket_timeout = int(os.getenv("MONGO_SOCKET_TIMEOUT", "120000"))
    
    # 환경 변수 확인
    if not all([mongo_user, mongo_password, mongo_host]):
        raise ValueError("필요한 MongoDB 환경 변수(MONGO_USER, MONGO_PASSWORD, MONGO_HOST)가 설정되지 않았습니다.")
    
    # 연결 문자열 구성
    connection_string = (
        f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}/{mongo_db}?"
        f"retryWrites=true&w=majority&connectTimeoutMS={mongo_connect_timeout}&socketTimeoutMS={mongo_socket_timeout}"
    )
    
    client = MongoClient(connection_string)
    db = client[mongo_db]
    
    # list_collection_names()를 사용하는 대신 날짜 범위로 컬렉션 이름 생성
    print("날짜 범위로 컬렉션 이름 생성 중...")
    
    # 최근 데이터를 가져오기 위한 날짜 범위 생성
    end_date = datetime.datetime.now()
    days_to_look_back = int(os.getenv("DAYS_TO_LOOK_BACK", "2200"))
    start_date = end_date - datetime.timedelta(days=days_to_look_back)
    
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("Date_%Y_%m_%d")
        date_list.append(date_str)
        current_date += datetime.timedelta(days=1)
    
    # 생성된 날짜 컬렉션 이름 중 실제로 존재하는지 확인
    df_list = []
    collection_count = 0
    
    for col_name in date_list:
        try:
            collection = db[col_name]
            # 해당 컬렉션이 존재하고 데이터가 있는지 확인
            count = collection.count_documents({}, limit=1)
            if count > 0:
                collection_count += 1
                data = list(collection.find())
                temp_df = pd.DataFrame(data)
                # MongoDB에서 자동 생성되는 '_id' 컬럼은 필요 없으므로 제거합니다.
                if '_id' in temp_df.columns:
                    temp_df.drop('_id', axis=1, inplace=True)
                # 날짜 정보 추가
                temp_df['date'] = col_name
                df_list.append(temp_df)
                print(f"컬렉션 {col_name}에서 {len(data)}개 데이터 로드됨")
        except Exception as e:
            print(f"컬렉션 {col_name} 처리 중 오류 발생: {str(e)}")
            continue
    
    print(f"총 {collection_count}개 컬렉션에서 데이터를 로드했습니다.")
    
    # 여러 DataFrame을 하나로 합칩니다.
    if df_list:
        df = pd.concat(df_list, axis=0, ignore_index=True)
        print(f"최종 데이터프레임 크기: {df.shape}")
    else:
        df = pd.DataFrame()
        print("로드된 데이터가 없습니다.")
    
    return df

def get_target_data(df, target_cols):
    """
    [원리 설명]
    - 주어진 DataFrame에서 특정 타겟 컬럼들만 추출하여 반환합니다.
    - 예를 들어, 유가 예측을 위한 'premiumGasoline', 'gasoline', 'diesel', 'kerosene' 컬럼을 추출할 때 사용됩니다.
    
    파라미터:
      - df: 원본 DataFrame
      - target_cols: 추출할 타겟 컬럼 리스트
    반환:
      - 타겟 컬럼들만 포함하는 DataFrame
    """
    return df[target_cols].copy()
