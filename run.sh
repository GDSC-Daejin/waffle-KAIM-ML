#!/bin/bash

# 스크립트 시작 시 로깅 활성화
set -x

# 현재 작업 디렉토리 출력
echo "현재 디렉토리: $(pwd)"

# 필요한 디렉토리 생성
mkdir -p ./cache ./results ./logs
echo "디렉토리 생성 완료"

# 현재 시간 가져오기
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "===== 유가 예측 시작 (${TIMESTAMP}) ====="

# 파이썬 가상환경이 있는지 확인
if [ -d "./venv" ]; then
    echo "가상환경 활성화 시도..."
    source ./venv/bin/activate || echo "가상환경 활성화 실패"
    which python
    python --version
else
    echo "가상환경이 없습니다. 시스템 Python 사용"
    which python
    python --version
fi

# main.py 파일 존재 확인
if [ ! -f "./main.py" ]; then
    echo "오류: main.py 파일을 찾을 수 없습니다"
    ls -la
    exit 1
fi

# 패키지 설치 확인
if ! python -c "import pandas, numpy, pymongo, torch" &> /dev/null; then
    echo "필수 패키지 설치 중..."
    pip install -r requirements.txt
fi

# CSV 파일 존재 확인
if [ ! -f "./cache/korea_economic_data.csv" ]; then
    echo "CSV 데이터 파일이 캐시 디렉토리에 없습니다. 복사 중..."
    if [ -f "./korea_economic_data.csv" ]; then
        cp ./korea_economic_data.csv ./cache/
        echo "CSV 파일 복사 완료"
    else
        echo "오류: korea_economic_data.csv 파일이 현재 디렉토리에 없습니다"
        exit 1
    fi
fi

# 메인 스크립트 실행
echo "main.py 실행 시도..."
python main.py

# 실행 결과 확인
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "===== 유가 예측 성공적으로 완료됨 ====="
    echo "결과 파일: ./results/final_forecast.json"
else
    echo "===== 유가 예측 실행 중 오류 발생 (코드: $EXIT_CODE) ====="
fi

# 가상환경 비활성화 (있는 경우)
if [ -d "./venv" ]; then
    deactivate
fi

# 디버깅 종료
set +x
