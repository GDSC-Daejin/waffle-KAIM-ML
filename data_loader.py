import pandas as pd
import re
from pymongo import MongoClient

def load_data_from_mongo():
    """
    MongoDB에서 'kaim' 데이터베이스 내 컬렉션 이름이 "Date_YYYY_MM_DD" 형식에 해당하는
    모든 컬렉션의 데이터를 불러와 하나의 DataFrame으로 반환합니다.
    
    하드코딩된 연결 정보를 사용하며, 연결 및 소켓 타임아웃을 2분(120,000ms)으로 설정합니다.
    아이디는 'kaim_r'이고, 호스트는 '152.70.233.5'입니다.
    """
    connection_string = (
        "mongodb://kaim_r:qwerty1234@152.70.233.5/kaim?"
        "retryWrites=true&w=majority&connectTimeoutMS=120000&socketTimeoutMS=120000"
    )
    db_name = "kaim"
    client = MongoClient(connection_string)
    db = client[db_name]
    
    # "Date_YYYY_MM_DD" 형식의 컬렉션 필터링
    all_collections = db.list_collection_names()
    pattern = r"^Date_\d{4}_\d{2}_\d{2}"
    selected_collections = [col for col in all_collections if re.match(pattern, col)]
    
    df_list = []
    for col_name in selected_collections:
        collection = db[col_name]
        data = list(collection.find())
        temp_df = pd.DataFrame(data)
        if '_id' in temp_df.columns:
            temp_df.drop('_id', axis=1, inplace=True)
        df_list.append(temp_df)
    
    if df_list:
        df = pd.concat(df_list, axis=0, ignore_index=True)
    else:
        df = pd.DataFrame()
    
    return df

def get_target_data(df, target_cols):
    """
    DataFrame에서 타겟 변수(예: 고급휘발유, 휘발유, 경유, 등유) 컬럼들만 추출하여 반환합니다.
    """
    return df[target_cols].copy()
