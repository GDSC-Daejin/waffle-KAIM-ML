import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from scipy.interpolate import make_interp_spline
import numpy as np
import mplcursors

def plot_results(dates, actual, future_dates, future_predictions):
    """
    실제 데이터와 미래 예측값을 하나의 그래프로 시각화합니다.
    
    :param dates: 과거 실제 데이터의 날짜 (datetime 시리즈)
    :param actual: 실제 값 (shape: (N, 4))
    :param future_dates: 미래 예측 날짜 (datetime 시리즈)
    :param future_predictions: 예측값 (shape: (future_steps, 4))
    """
    plt.figure(figsize=(14, 8))
    # 4가지 연료(영어)
    fuel_types = ['premiumGasoline', 'gasoline', 'diesel', 'kerosene']
    ax = plt.gca()
    
    # 실제 데이터 플롯
    for i, fuel in enumerate(fuel_types):
        ax.plot(dates, actual[:, i], label=f"{fuel} (actual)", marker='o', linestyle='-')
    
    # 미래 예측 (스플라인 보간)
    future_dates_num = mdates.date2num(future_dates)
    for i, fuel in enumerate(fuel_types):
        y_future = future_predictions[:, i]
        x_new = np.linspace(future_dates_num.min(), future_dates_num.max(), 300)
        spline = make_interp_spline(future_dates_num, y_future, k=3)
        y_smooth = spline(x_new)
        x_new_dates = mdates.num2date(x_new)
        line_future, = ax.plot(x_new_dates, y_smooth, label=f"{fuel} (predicted)", linestyle='--', linewidth=2)
        cursor = mplcursors.cursor(line_future, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            x_val = sel.target[0]
            y_val = sel.target[1]
            date_str = mdates.num2date(x_val).strftime('%Y-%m-%d')
            sel.annotation.set(text=f"{fuel}\n{date_str}\n{y_val:.2f}")
            sel.annotation.get_bbox_patch().set(alpha=0.7)
    
    plt.title("Fuel Price Prediction", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.2f}'))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
