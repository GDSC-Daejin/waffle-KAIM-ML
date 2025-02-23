import torch
import numpy as np

def make_predictions(model, X):
    model.eval()
    with torch.no_grad():
        return model(X).numpy() 

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))
