import pandas as pd
import re
from pymongo import MongoClient

def load_data_from_mongo():
    """
    [원리 설명]
    - MongoDB 서버에 저장된 'kaim' 데이터베이스에서 데이터를 가져옵니다.
    - 각 날짜별로 저장된 컬렉션의 이름은 "Date_YYYY_MM_DD" 형식을 따릅니다.
      예를 들어, "Date_2025_01_23", "Date_2025_02_05" 등이 이에 해당합니다.
    - 이 함수는 해당 형식의 모든 컬렉션을 찾아 각 컬렉션의 데이터를 DataFrame으로 변환한 후,
      이를 하나의 DataFrame으로 합쳐 반환합니다.
    - 데이터베이스 연결은 하드코딩된 연결 문자열(아이디 'kaim_r', 호스트 '152.70.233.5')을 사용합니다.
    
    [주요 기능]
    - MongoClient를 이용해 MongoDB에 연결합니다.
    - 모든 컬렉션 이름을 가져와 정규표현식으로 "Date_YYYY_MM_DD" 패턴에 맞는 컬렉션만 선택합니다.
    - 각 컬렉션의 데이터를 pandas DataFrame으로 변환하고, 불필요한 '_id' 컬럼은 제거합니다.
    - 여러 DataFrame을 하나로 통합하여 반환합니다.
    """
    connection_string = (
        "mongodb://kaim_r:qwerty1234@152.70.233.5/kaim?"
        "retryWrites=true&w=majority&connectTimeoutMS=120000&socketTimeoutMS=120000"
    )
    db_name = "kaim"
    client = MongoClient(connection_string)
    db = client[db_name]
    
    # 모든 컬렉션 이름 중 "Date_YYYY_MM_DD" 패턴에 맞는 컬렉션만 선택합니다.
    all_collections = db.list_collection_names()
    pattern = r"^Date_\d{4}_\d{2}_\d{2}"
    selected_collections = [col for col in all_collections if re.match(pattern, col)]
    
    df_list = []
    # 선택된 각 컬렉션의 데이터를 DataFrame으로 변환합니다.
    for col_name in selected_collections:
        collection = db[col_name]
        data = list(collection.find())
        temp_df = pd.DataFrame(data)
        # MongoDB에서 자동 생성되는 '_id' 컬럼은 필요 없으므로 제거합니다.
        if '_id' in temp_df.columns:
            temp_df.drop('_id', axis=1, inplace=True)
        df_list.append(temp_df)
    
    # 여러 DataFrame을 하나로 합칩니다.
    if df_list:
        df = pd.concat(df_list, axis=0, ignore_index=True)
    else:
        df = pd.DataFrame()
    
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
