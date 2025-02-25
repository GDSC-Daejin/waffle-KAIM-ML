import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# data_loader.py의 load_data_from_mongo()를 import하여 MongoDB 데이터를 불러옵니다.
from data_loader import load_data_from_mongo
# model.py의 LSTMModel 클래스를 import하여 LSTM 모델을 구성합니다.
from model import LSTMModel
# train.py의 train_model() 함수를 import하여 모델을 학습합니다.
from train import train_model
# predict.py의 make_predictions()와 calculate_rmse() 함수를 import하여 예측 및 평가를 수행합니다.
from predict import make_predictions, calculate_rmse
# visualize.py의 plot_results() 함수를 import하여 예측 결과를 시각화합니다.
from visualize import plot_results

def parse_list_to_float(val):
    """
    [원리 설명]
    - 데이터베이스에서 불러온 값 중 일부가 리스트 형태(예: [1597.04, 1679.35, ...])로 저장될 수 있습니다.
    - 이 함수는 리스트 내 숫자들을 float로 변환한 후 평균값을 계산하여 단일 숫자로 반환합니다.
    - 만약 변환 가능한 값이 없으면 np.nan을 반환합니다.
    """
    if isinstance(val, list):
        numeric_vals = []
        for item in val:
            try:
                numeric_vals.append(float(item))
            except:
                pass
        if len(numeric_vals) == 0:
            return np.nan
        else:
            return float(np.mean(numeric_vals))
    else:
        try:
            return float(val)
        except:
            return np.nan

def create_dataset_for_XY(X_data, Y_data, look_back=3):
    """
    [원리 설명]
    - 시계열 예측을 위해, 연속된 look_back 기간의 데이터를 하나의 입력(X)으로,
      그 다음 시점의 데이터를 타겟(Y)으로 구성하는 데이터셋을 생성합니다.
    - 예를 들어, look_back=3이면 인덱스 0~2를 X, 3번째 데이터를 Y로 구성합니다.
    
    반환:
      - X: (샘플수, look_back, 입력 특성 수) 형태의 numpy 배열
      - Y: (샘플수, 타겟 특성 수) 형태의 numpy 배열
    """
    X, Y = [], []
    for i in range(len(X_data) - look_back):
        X.append(X_data[i : i + look_back])
        Y.append(Y_data[i + look_back])
    return np.array(X), np.array(Y)

def main():
    # ------------------------------------------------------------------
    # 1) MongoDB에서 데이터 불러오기
    # ------------------------------------------------------------------
    # load_data_from_mongo()를 호출하여, MongoDB의 'kaim' 데이터베이스에 저장된
    # 여러 날짜별 컬렉션 데이터를 하나의 DataFrame으로 통합합니다.
    df = load_data_from_mongo()

    # ------------------------------------------------------------------
    # 2) 날짜 전처리
    # ------------------------------------------------------------------
    # 'date' 컬럼은 "Date_YYYY_MM_DD" 형식의 문자열이므로, "Date_" 접두사를 제거하고,
    # "%Y_%m_%d" 포맷으로 datetime 객체로 변환합니다.
    df['date'] = pd.to_datetime(
        df['date'].str.replace("Date_", "", regex=False),
        format="%Y_%m_%d",
        errors='coerce'
    )
    df.sort_values('date', inplace=True)
    # (선택) 날짜 보정을 위해 1일을 빼줍니다.
    df['date'] = df['date'] - pd.Timedelta(days=1)

    # ------------------------------------------------------------------
    # 3) 특징(Features) 데이터 추출 및 리스트 데이터 변환
    # ------------------------------------------------------------------
    # 'date' 컬럼은 학습에 사용하지 않으므로 제거하고, 나머지 모든 컬럼을 features로 사용합니다.
    # 이 features에는 경제 지표와 원유 관련 정보가 모두 포함되어 있습니다.
    features = df.drop(columns=['date'])
    # 리스트 형태 데이터(예: 일부 유가 데이터)는 parse_list_to_float()를 통해 단일 평균값으로 변환합니다.
    for col in features.columns:
        features[col] = features[col].apply(parse_list_to_float)
    # 모든 데이터를 숫자로 변환하고, 변환에 실패한(전체가 NaN) 컬럼은 제거합니다.
    features = features.apply(pd.to_numeric, errors='coerce')
    features = features.dropna(axis=1, how='all')

    # ------------------------------------------------------------------
    # 4) 입력(Features)과 타겟(Target) 분리 및 변수별 활용
    # ------------------------------------------------------------------
    # 타겟 컬럼: 예측하고자 하는 유가 관련 4개 변수
    #   - premiumGasoline: 고급 휘발유 가격 (품질에 따른 차별화)
    #   - gasoline: 일반 휘발유 가격
    #   - diesel: 경유 가격 (상업 및 산업용)
    #   - kerosene: 등유 가격 (난방 등 특정 용도)
    #
    # 나머지 컬럼들은 입력 변수로 사용되며, 각 변수는 아래와 같이 활용됩니다:
    #   - 'Dubai_Val', 'Brent_Val', 'WTI_Val': 국제 원유 가격 지표로, 전 세계 유가 동향을 반영합니다.
    #   - 'KRW_Rating': 원/달러 환율로, 환율 변동이 원유 수입 비용에 영향을 미칩니다.
    #   - 'cpi_year', 'cpi': 소비자물가지수로, 물가 상승이 유가에 간접적 영향을 줍니다.
    #   - 'ppi_total_year', 'ppi_total': 생산자물가지수로, 생산비용 변동을 반영합니다.
    #   - 'per_capita_gni_nominal', 'per_capita_gni_nominal_year': 1인당 국민총소득 지표로 경제 규모를 나타냅니다.
    #   - 'per_capita_gdp_nominal', 'per_capita_gdp_nominal_year': 1인당 GDP 지표로 경제 활황 정도를 나타냅니다.
    #   - 'gdp_growth_rate', 'gdp_growth_rate_year': GDP 성장률로, 경제 성장 추세를 반영합니다.
    #   - 'interest_rate': 금리로, 금융 환경 변화가 에너지 수요에 영향을 줍니다.
    #   - 'area': 지역 정보로, 특정 지역의 경제 상황이나 유통 구조 차이를 고려할 수 있습니다.
    target_cols = ["premiumGasoline", "gasoline", "diesel", "kerosene"]
    for col in target_cols:
        if col not in features.columns:
            raise ValueError(f"Target column '{col}' does not exist in the data.")
    all_cols = list(features.columns)
    input_cols = [c for c in all_cols if c not in target_cols]

    # ------------------------------------------------------------------
    # 5) 데이터 정규화 및 배열 변환
    # ------------------------------------------------------------------
    # 입력 데이터(X_data)와 타겟 데이터(Y_data)를 각각 NumPy 배열로 변환한 후,
    # MinMaxScaler를 사용하여 0과 1 사이의 값으로 정규화합니다.
    X_data = features[input_cols].values.astype(float)
    Y_data = features[target_cols].values.astype(float)

    scaler_input = MinMaxScaler()
    X_data_scaled = scaler_input.fit_transform(X_data)

    scaler_target = MinMaxScaler()
    Y_data_scaled = scaler_target.fit_transform(Y_data)

    # ------------------------------------------------------------------
    # 6) 시계열 데이터셋 생성
    # ------------------------------------------------------------------
    # look_back 기간(예: 3일)을 사용해, 과거 데이터를 하나의 샘플로 묶고,
    # 그 다음 날의 데이터를 타겟으로 설정하는 시계열 데이터셋을 생성합니다.
    look_back = 3
    X_np, Y_np = create_dataset_for_XY(X_data_scaled, Y_data_scaled, look_back=look_back)
    X_torch = torch.FloatTensor(X_np)
    Y_torch = torch.FloatTensor(Y_np)

    # ------------------------------------------------------------------
    # 7) LSTM 모델 구성 및 학습
    # ------------------------------------------------------------------
    # LSTM 모델은 입력 시퀀스의 패턴을 학습하여 미래 타겟 값을 예측합니다.
    # 파라미터:
    #   - input_dim: 입력 데이터의 특성 수 (예: 위의 경제 지표 및 원유 지표 총 20개 정도)
    #   - hidden_dim: LSTM 내부 은닉 상태의 차원 (중요 정보를 담는 벡터 크기)
    #   - layer_dim: LSTM 레이어의 수
    #   - output_dim: 예측할 타겟 변수의 수 (여기서는 4)
    input_dim = X_torch.shape[2]
    hidden_dim = 50
    layer_dim = 1
    output_dim = Y_torch.shape[1]  # 4개 타겟

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    # train_model() 함수 (train.py)를 호출하여 모델 학습을 진행합니다.
    model = train_model(model, X_torch, Y_torch, epochs=100, batch_size=32)

    # ------------------------------------------------------------------
    # 8) 예측 및 평가
    # ------------------------------------------------------------------
    # 학습된 모델을 사용해 입력 데이터에 대한 예측을 수행한 후,
    # scaler_target을 통해 정규화된 예측값을 원래 스케일로 복원합니다.
    # 그리고 실제 타겟 값과 예측 값의 차이를 RMSE로 평가합니다.
    preds_scaled = make_predictions(model, X_torch)
    preds = scaler_target.inverse_transform(preds_scaled)
    Y_actual_scaled = Y_torch.numpy()
    Y_actual = scaler_target.inverse_transform(Y_actual_scaled)
    rmse = calculate_rmse(Y_actual, preds)
    print(f"RMSE: {rmse:.2f}")

    # ------------------------------------------------------------------
    # 9) 미래 7일치 예측
    # ------------------------------------------------------------------
    # 마지막 look_back 기간의 입력 데이터를 기반으로 미래 7일치 타겟 값을 예측합니다.
    # (주의: 이 부분은 단순 슬라이딩 방식을 사용하므로, 입력과 출력의 차원 불일치 문제 발생 가능)
    future_steps = 7
    last_seq = X_data_scaled[-look_back:]

    model.eval()
    future_preds_scaled = []
    with torch.no_grad():
        current_seq = last_seq.copy()
        for _ in range(future_steps):
            inp = torch.FloatTensor(current_seq).unsqueeze(0)  # (1, look_back, input_dim)
            pred_s = model(inp).numpy()[0]  # 예측 결과: 타겟 4개
            future_preds_scaled.append(pred_s)
            # 슬라이딩: 현재 시퀀스의 첫 행을 제거하고, 마지막 행(예측값)을 추가하여 새로운 시퀀스를 구성
            current_seq = np.vstack([current_seq[1:], current_seq[-1]])
    
    future_preds_scaled = np.array(future_preds_scaled)
    future_preds = scaler_target.inverse_transform(future_preds_scaled)

    # ------------------------------------------------------------------
    # 10) 미래 예측 날짜 생성 및 결과 출력
    # ------------------------------------------------------------------
    # 마지막 실제 날짜 이후 7일의 날짜를 생성하고, 각 날짜별 미래 예측값을 콘솔에 출력합니다.
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)

    print("\nFuture 7-day Predictions:")
    for d, vals in zip(future_dates, future_preds):
        print(f"{d.date()} -> premiumGasoline={vals[0]:.2f}, gasoline={vals[1]:.2f}, diesel={vals[2]:.2f}, kerosene={vals[3]:.2f}")

    # ------------------------------------------------------------------
    # 11) 시각화
    # ------------------------------------------------------------------
    # plot_results() 함수를 호출하여, 학습에 사용된 실제 타겟 데이터와 미래 예측 결과를 그래프로 시각화합니다.
    used_dates = df['date'].iloc[look_back:]
    hist_data = Y_data[-(len(used_dates)):]
    hist_data_inv = scaler_target.inverse_transform(hist_data)

    plot_results(used_dates, hist_data_inv, future_dates, future_preds)

    # ------------------------------------------------------------------
    # [프로젝트 요약]
    # ------------------------------------------------------------------
    # 이 프로젝트는 MongoDB에서 불러온 경제 및 원유 관련 데이터를 기반으로
    # LSTM 시계열 모델을 학습하여 미래 유가(예: premiumGasoline, gasoline, diesel, kerosene)를 예측합니다.
    # 
    # 주요 단계:
    #  1) MongoDB에서 여러 날짜별 데이터를 하나의 DataFrame으로 통합.
    #  2) 'date' 컬럼을 파싱하여 정렬 및 필요 시 보정.
    #  3) 리스트 형태 데이터(예: diesel, kerosene)의 평균값을 구해 단일 float 값으로 변환.
    #  4) 다양한 경제 지표 및 원유 관련 변수들('Dubai_Val', 'Brent_Val', 'WTI_Val', 'KRW_Rating', 
    #     'cpi_year', 'cpi', 'ppi_total_year', 'ppi_total', 'per_capita_gni_nominal', 'per_capita_gni_nominal_year',
    #     'per_capita_gdp_nominal', 'per_capita_gdp_nominal_year', 'gdp_growth_rate', 'gdp_growth_rate_year',
    #     'interest_rate', 'area')를 입력(Features)으로 사용.
    #     - 이들 변수는 LSTM 모델의 입력 벡터에 포함되어, 각 변수의 가중치는 학습을 통해 자동으로 조정됩니다.
    #  5) 타겟 변수는 유가 관련 4개 변수로 설정.
    #  6) 데이터를 MinMaxScaler로 정규화하고, 시계열 데이터셋을 생성.
    #  7) LSTM 모델을 학습하여 입력 시퀀스의 패턴을 기반으로 미래 유가를 예측.
    #  8) 학습 데이터에 대한 예측 성능은 RMSE로 평가.
    #  9) 마지막 데이터를 기반으로 미래 7일치 예측을 수행하고, 결과를 콘솔 출력 및 그래프로 시각화.
    #
    # 핵심 아이디어:
    # - 여러 경제 및 원유 지표를 함께 고려하여 복합적인 시계열 패턴을 학습하도록 LSTM 모델을 구성.
    # - 모델은 각 입력 변수의 영향을 내포하는 가중치 행렬을 학습하며, 이 가중치는 학습 후에 최적화되어 미래 유가 예측에 활용됩니다.
    # ------------------------------------------------------------------

if __name__ == "__main__":
    main()
