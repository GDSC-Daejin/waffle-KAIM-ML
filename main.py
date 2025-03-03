import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_loader import load_data_from_mongo
from data_preprocessor import create_dataset, prepare_data
from model import LSTMModel
from train import train_model
from visualize import plot_results

def flatten(x):
    if isinstance(x, list):
        flat_list = []
        for item in x:
            flat_list.extend(flatten(item))
        return flat_list
    else:
        return [x]

def flatten_and_average(x):
    if isinstance(x, list):
        flat_list = flatten(x)
        numeric_vals = []
        for item in flat_list:
            try:
                numeric_vals.append(float(item))
            except:
                pass
        return np.mean(numeric_vals) if numeric_vals else np.nan
    else:
        try:
            return float(x)
        except:
            return np.nan

def flatten_to_string(x):
    if isinstance(x, list):
        return ",".join(str(v) for v in flatten(x))
    return str(x)

def predict_future_exo(model, last_sequence, future_steps, target_scaler, target_start_index):
    model.eval()
    predictions = []
    current_seq = last_sequence.copy()  # shape: (look_back, num_features)
    date_index = 0  # 'date' 열을 0번째라고 가정
    for _ in range(future_steps):
        input_tensor = torch.FloatTensor(current_seq).unsqueeze(0)
        with torch.no_grad():
            pred = model(input_tensor).numpy()[0]
        predictions.append(pred)
        
        # 슬라이딩 윈도우
        new_row = current_seq[-1].copy()
        new_row[date_index] += 1  # 날짜 ordinal 값 +1
        new_row[target_start_index : target_start_index + len(pred)] = pred
        new_row = new_row.reshape(1, -1)
        current_seq = np.vstack([current_seq[1:], new_row])

    predictions = np.array(predictions)
    predictions = target_scaler.inverse_transform(predictions)
    return predictions

def predict_for_df(df_in, all_features, target_cols, look_back=3, future_steps=7, epochs=100):
    df_in = df_in.sort_values('date').copy()

    features = df_in[all_features].copy()

    if 'date' in features.columns:
        features['date'] = features['date'].apply(lambda x: x.toordinal() if pd.notnull(x) else np.nan)
    if 'area' in features.columns:
        # area를 숫자로 변환
        features['area'] = pd.factorize(features['area'])[0]

    for col in features.columns:
        if col not in ['date','area']:
            features[col] = features[col].apply(flatten_and_average)
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors='coerce')

    # 전부 NaN인 열/행 제거
    features.dropna(axis=1, how='all', inplace=True)
    features.dropna(axis=0, how='all', inplace=True)
    if len(features) == 0:
        print("[WARN] predict_for_df: after dropna, no data left.")
        return pd.DataFrame()

    # 스케일러
    f_scaler = MinMaxScaler()
    norm_feat = f_scaler.fit_transform(features.values)

    # 타겟 스케일러
    t_scaler = MinMaxScaler()
    valid_tcols = [c for c in target_cols if c in df_in.columns]
    if not valid_tcols:
        print("[WARN] no valid target columns in df_in.")
        return pd.DataFrame()
    
    tdata = df_in[valid_tcols].fillna(0).values
    t_scaler.fit(tdata)

    # 시계열 생성
    X, Y_full = create_dataset(norm_feat, look_back)
    if len(X) < 1:
        print("[WARN] not enough data for look_back.")
        return pd.DataFrame()
    X, Y_full = prepare_data(X, Y_full)

    # 타겟 인덱스
    col_list = list(features.columns)
    target_indices = []
    for tc in valid_tcols:
        if tc in col_list:
            target_indices.append(col_list.index(tc))
    if not target_indices:
        print("[WARN] target indices empty.")
        return pd.DataFrame()

    Y_np = Y_full.numpy()
    Y = Y_np[:, target_indices]

    input_dim = X.shape[2]
    hidden_dim = 50
    layer_dim = 1
    output_dim = len(target_indices)

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    model = train_model(model, X, torch.FloatTensor(Y), epochs=epochs)

    last_seq = norm_feat[-look_back:]
    future_preds = predict_future_exo(model, last_seq, future_steps, t_scaler, min(target_indices))

    # 날짜 생성
    last_date = df_in['date'].iloc[-1]
    if pd.isnull(last_date):
        print("[WARN] last_date is NaN.")
        return pd.DataFrame()

    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)
    pred_df = pd.DataFrame(future_preds, columns=valid_tcols)
    pred_df['date'] = future_dates

    return pred_df

def main():
    df = load_data_from_mongo()
    
    df['date'] = df['date'].str.replace("Date_", "", regex=False)
    df['date'] = pd.to_datetime(df['date'], format="%Y_%m_%d", errors='coerce')
    df.sort_values('date', inplace=True)

    # 만약 area가 여러 지역을 쉼표로 묶어서 가지고 있다면, explode 시켜야 함
    # 예: "National,Seoul,Busan" => ["National","Seoul","Busan"] => 3행
    df['area'] = df['area'].fillna("Unknown").apply(flatten_to_string)
    # explode를 위해 split
    df['area'] = df['area'].apply(lambda x: x.split(',') if isinstance(x,str) and ',' in x else [x])
    # explode
    df = df.explode('area')
    # strip해서 공백 제거
    df['area'] = df['area'].apply(lambda x: x.strip() if isinstance(x,str) else x)

    # 다른 컬럼 -> flatten_and_average
    for col in df.columns:
        if col not in ['date','area']:
            df[col] = df[col].apply(flatten_and_average)
    # NaN 처리
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)

    # 전체 피처
    # 원하는 공통지표/유가/등등 실제 컬럼명 확인 후 맞춰야 함
    all_features = df.columns.tolist()
    # date, area를 맨 앞으로
    if 'date' in all_features:
        all_features.remove('date')
        all_features.insert(0,'date')
    if 'area' in all_features:
        all_features.remove('area')
        all_features.insert(1,'area')

    # 예측할 타겟 (유가 4종이라고 가정)
    target_cols = ['gasoline','premiumGasoline','diesel','kerosene']

    ##########################
    # 1) 전국 평균 예측
    ##########################
    # date만으로 groupby -> 모든 숫자 평균
    # as_index=False -> date컬럼 유지
    nat_gb = df.groupby('date', as_index=False).mean(numeric_only=True)
    nat_gb['area'] = 'National'
    nat_pred = predict_for_df(nat_gb, all_features, target_cols, look_back=3, future_steps=7)
    # (빈 df일 수도 있음)

    ##########################
    # 2) 지역별 예측
    ##########################
    # date+area로 groupby -> 평균
    region_gb = df.groupby(['date','area'], as_index=False).mean(numeric_only=True)
    # 이제 unique area들에 대해 예측
    unique_areas = region_gb['area'].unique()

    region_preds_list = []
    for reg in unique_areas:
        if reg == 'National':
            continue
        sub_df = region_gb[region_gb['area'] == reg].copy()
        while len(sub_df) < 3:
            sub_df = pd.concat([sub_df, sub_df.iloc[[-1]]], ignore_index=True)
        r_pred = predict_for_df(sub_df, all_features, target_cols, look_back=3, future_steps=7)
        if not r_pred.empty:
            r_pred['area'] = reg
            region_preds_list.append(r_pred)

    ##########################
    # 3) 콘솔 출력
    ##########################
    print("\n=== 7일치 유가 예측 결과 ===\n")

    # 전국 평균
    print("[전국 평균 예측]\n")
    if not nat_pred.empty:
        print("지역: National")
        for _, row in nat_pred.iterrows():
            d = row['date']
            date_str = d.strftime('%Y-%m-%d') if pd.notnull(d) else "Unknown"
            print(f"{date_str}: gasoline={row.get('gasoline',np.nan):.2f}, "
                  f"premiumGasoline={row.get('premiumGasoline',np.nan):.2f}, "
                  f"diesel={row.get('diesel',np.nan):.2f}, "
                  f"kerosene={row.get('kerosene',np.nan):.2f}")
    else:
        print("전국 평균 예측 데이터가 없습니다.\n")

    # 지역별
    print("\n[지역별 예측]\n")
    if not region_preds_list:
        print("지역별 예측 결과가 없습니다(데이터가 없거나 NaN).")
    else:
        for rp in region_preds_list:
            region_name = rp['area'].iloc[0]
            print(f"지역: {region_name}")
            for _, row in rp.iterrows():
                d = row['date']
                date_str = d.strftime('%Y-%m-%d') if pd.notnull(d) else "Unknown"
                print(f"{date_str}: gasoline={row.get('gasoline',np.nan):.2f}, "
                      f"premiumGasoline={row.get('premiumGasoline',np.nan):.2f}, "
                      f"diesel={row.get('diesel',np.nan):.2f}, "
                      f"kerosene={row.get('kerosene',np.nan):.2f}")
            print()

    ##########################
    # 4) 시각화
    ##########################
    # (전국 평균 기반)
    if not nat_pred.empty:
        # groupby된 nat_gb 가 실제값
        actual_dates = nat_gb['date']
        actual_data = nat_gb[target_cols].fillna(0).values
        future_dates = nat_pred['date']
        future_data = nat_pred[target_cols].fillna(0).values
        plot_results(actual_dates, actual_data, future_dates, future_data)

if __name__ == "__main__":
    main()
