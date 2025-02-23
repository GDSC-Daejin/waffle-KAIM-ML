import torch
import pandas as pd
from data_loader import load_data, get_price_data
from data_preprocessor import normalize_data, create_dataset, prepare_data
from model import LSTMModel
from train import train_model
from predict import make_predictions, calculate_rmse
from visualize import plot_results

def main():
    # 데이터 로드
    df = load_data('Date_*.csv')
    price_data = get_price_data(df)

    # 날짜 열을 인덱스로 설정
    df['date'] = pd.to_datetime(df['date'].str.split('_').str[1:4].str.join('-'))
    price_data.index = df['date']
    price_data.sort_index(inplace=True)

    # 데이터 전처리
    normalized_data, scaler = normalize_data(price_data.values)
    
    # 데이터셋 생성
    look_back = 3
    X, Y = create_dataset(normalized_data, look_back)
    X, Y = prepare_data(X, Y)

    # 모델 구축 및 훈련
    input_dim = X.shape[2]
    hidden_dim = 50
    layer_dim = 1
    output_dim = Y.shape[1]
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    model = train_model(model, X, Y)

    # 예측 수행
    predictions = make_predictions(model, X)

    # 역정규화
    predictions = scaler.inverse_transform(predictions)
    Y = scaler.inverse_transform(Y.numpy())

    # RMSE 계산
    rmse = calculate_rmse(Y, predictions)
    print(f'RMSE: {rmse:.2f}')

    # 결과 시각화
    plot_results(price_data.index[look_back:], Y, predictions)
    
if __name__ == "__main__":
    main()
