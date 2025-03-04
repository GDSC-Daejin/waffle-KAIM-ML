from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# 데이터 불러오기
file_path = "korea_economic_data.csv"
df = pd.read_csv(file_path)

# 데이터 타입 확인
print(df.dtypes)

# 숫자가 아닌 열이 있는지 확인
non_numeric_cols = df.select_dtypes(include=['object']).columns
print("문자열 포함된 열:", non_numeric_cols)

# 숫자만 있는 데이터프레임 만들기 (문자열 열 제거)
df_numeric = df.drop(columns=non_numeric_cols)

# 스케일러 정의
scaler = MinMaxScaler()

# 정규화 적용
scaled_data = scaler.fit_transform(df_numeric)

# 데이터프레임으로 변환
df_scaled = pd.DataFrame(scaled_data, columns=df_numeric.columns)

# 원래 도시명 컬럼 다시 추가
df_scaled.insert(0, "City", df["City"])  

print(df_scaled.head())  # 변환된 데이터 확인
