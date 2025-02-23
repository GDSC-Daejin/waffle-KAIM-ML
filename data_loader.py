import pandas as pd
import glob
import ast 

def load_data(file_pattern):
    """여러 CSV 파일을 로드하고 병합합니다."""
    all_files = glob.glob(file_pattern)
    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        df_list.append(df)
    return pd.concat(df_list, axis=0, ignore_index=True)

def get_price_data(df):
    """데이터프레임에서 가격 데이터를 추출합니다."""
    price_data = df[['Dubai_Val', 'Brent_Val', 'WTI_Val', 'gasoline', 'diesel', 'kerosene']].copy()
    for col in ['gasoline', 'diesel', 'kerosene']:
        price_data.loc[:, col] = price_data[col].apply(lambda x: ast.literal_eval(x)[0])
    return price_data

