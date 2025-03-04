import os
import pandas as pd

# 현재 실행 중인 파일의 폴더 위치 가져오기
current_dir = os.path.dirname(os.path.abspath(__file__))

# CSV 파일이 존재하는 폴더에 맞춰서 경로 설정
file_path = os.path.join(current_dir, "korea_economic_data.csv")

# 파일이 존재하는지 확인
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ 파일이 존재하지 않습니다: {file_path}")

# CSV 파일 로드
df = pd.read_csv(file_path)
print("✅ 데이터 로드 성공!")
