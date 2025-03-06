import torch
import numpy as np

def ensemble_predict_future(ensemble_model, last_sequence, future_steps, scaler, target_indices=None):
    """
    앙상블 모델을 사용하여 미래 예측 수행
    """
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(future_steps):
        input_tensor = torch.FloatTensor(current_seq).unsqueeze(0)
        with torch.no_grad():
            pred = ensemble_model(input_tensor).numpy()[0]
        predictions.append(pred)
        
        current_seq = np.vstack([current_seq[1:], pred])
    
    predictions = np.array(predictions)
    
    if target_indices:
        full_predictions = np.zeros((future_steps, current_seq.shape[1]))
        for i, pred in enumerate(predictions):
            full_predictions[i, target_indices] = pred
        full_predictions = scaler.inverse_transform(full_predictions)
        return full_predictions[:, target_indices]
    else:
        return scaler.inverse_transform(predictions)
