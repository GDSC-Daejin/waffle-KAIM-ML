# first line: 112
@memory.cache
def create_features(df, target_col):
    print("\n=== 특성 엔지니어링 시작 ===")
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 예시: 간단한 이동평균, EMA, lag 변수 생성 (추가 특성은 필요에 따라 확대)
    for feature in [target_col]:
        for window in [3, 7, 14]:
            df[f'{feature}_MA{window}'] = df[target_col].rolling(window=window).mean()
        for lag in [1, 2, 3]:
            df[f'{feature}_lag{lag}'] = df[target_col].shift(lag)
    
    df = df.dropna()
    print(f"특성 생성 후 데이터 크기: {df.shape}")
    print("=== 특성 엔지니어링 완료 ===")
    return df
