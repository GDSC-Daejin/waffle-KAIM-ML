import matplotlib.pyplot as plt

def plot_results(dates, actual, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual')
    plt.plot(dates, predictions, label='Predictions')
    plt.title('Oil Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
