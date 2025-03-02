# first line: 70
@memory.cache
def load_and_preprocess_data(file_path):
    logger.info(f"데이터 로드 중: {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"원본 데이터 크기: {df.shape}")
    
    # 날짜 처리: "Date_" 접두어 제거 후 정렬
    df['date'] = pd.to_datetime(df['date'].str.replace('Date_', ''), format='%Y_%m_%d')
    df = df.sort_values('date').reset_index(drop=True)
    
    # 지역별 연료 가격 데이터 처리 (문자열 형태의 리스트를 실제 값으로 변환)
    regions = ['National', 'Seoul', 'Busan', 'Daegu', 'Incheon', 'Gwangju', 'Daejeon', 'Ulsan',
               'Sejong', 'Gyeonggi', 'Gangwon', 'Chungbuk', 'Chungnam', 'Jeonbuk', 'Jeonnam',
               'Gyeongbuk', 'Gyeongnam']
    for fuel_type in ['gasoline', 'premiumGasoline', 'diesel', 'kerosene']:
        if fuel_type in df.columns and isinstance(df[fuel_type].iloc[0], str):
            df[fuel_type] = df[fuel_type].apply(lambda x: eval(x) if isinstance(x, str) else x)
            if isinstance(df[fuel_type].iloc[0], list) and len(df[fuel_type].iloc[0]) == len(regions):
                for i, region in enumerate(regions):
                    df[f'{fuel_type}_{region}'] = df[fuel_type].apply(lambda x: float(x[i]) if isinstance(x, list) else np.nan)
            df = df.drop(fuel_type, axis=1)
    if 'area' in df.columns and isinstance(df['area'].iloc[0], str):
        df = df.drop('area', axis=1)
    
    # 자료형 최적화
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    
    # 수치형 컬럼 및 유종별 열 추출
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    fuel_types = ['gasoline', 'premiumGasoline', 'diesel', 'kerosene']
    fuel_columns = {}
    for ft in fuel_types:
        fuel_columns[ft] = [col for col in df.columns if col.startswith(f"{ft}_")]
    
    logger.info(f"예측 대상 변수: {fuel_columns}")
    logger.info(f"데이터 기간: {df['date'].min().strftime('%Y-%m-%d')} 부터 {df['date'].max().strftime('%Y-%m-%d')}, 총 {df.shape[0]}일")
    return df, numeric_cols, fuel_columns
