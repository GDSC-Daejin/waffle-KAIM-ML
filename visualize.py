import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from scipy.interpolate import make_interp_spline
import numpy as np
import mplcursors

def plot_results(dates, actual, future_dates, future_predictions):
    """
    실제 가격과 미래 예측 결과를 함께 시각화합니다.
    실제 데이터는 원본 값으로, 미래 예측은 스플라인 보간을 통해 부드러운 선으로 표시됩니다.
    마우스 오버 시 해당 날짜와 가격이 툴팁으로 표시됩니다.
    
    :param dates: 실제 데이터의 날짜 (datetime 시리즈)
    :param actual: 실제 가격 데이터 (2D 배열, 순서: 고급휘발유, 휘발유, 경유, 등유)
    :param future_dates: 미래 예측 날짜 (datetime 시리즈)
    :param future_predictions: 미래 예측 데이터 (2D 배열)
    """
    plt.figure(figsize=(14, 8))
    fuel_types = ['고급휘발유', '휘발유', '경유', '등유']
    ax = plt.gca()
    
    # 실제 데이터 플롯
    for i, fuel in enumerate(fuel_types):
        ax.plot(dates, actual[:, i], label=f"{fuel} 실제값", marker='o', linestyle='-')
    
    # 미래 예측 데이터 플롯 (스플라인 보간)
    future_dates_num = mdates.date2num(future_dates)
    for i, fuel in enumerate(fuel_types):
        y_future = future_predictions[:, i]
        x_new = np.linspace(future_dates_num.min(), future_dates_num.max(), 300)
        spline = make_interp_spline(future_dates_num, y_future, k=3)
        y_smooth = spline(x_new)
        x_new_dates = mdates.num2date(x_new)
        line_future, = ax.plot(x_new_dates, y_smooth, label=f"{fuel} 예측값", linestyle='--', linewidth=2)
        cursor = mplcursors.cursor(line_future, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            x_val = sel.target[0]
            y_val = sel.target[1]
            date_str = mdates.num2date(x_val).strftime('%Y-%m-%d')
            sel.annotation.set(text=f"{fuel}\n{date_str}\n{y_val:.2f}")
            sel.annotation.get_bbox_patch().set(alpha=0.7)
    
    plt.title("유가 예측 결과", fontsize=16)
    plt.xlabel("날짜", fontsize=14)
    plt.ylabel("가격", fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.2f}'))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
