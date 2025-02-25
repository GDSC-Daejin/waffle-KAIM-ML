import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, X_train, Y_train, epochs=100, batch_size=32):
    """
    LSTM 모델을 학습하는 함수입니다.
    :param model: 학습할 모델
    :param X_train: 입력 데이터
    :param Y_train: 타겟 데이터
    :param epochs: 전체 에포크 수
    :param batch_size: 배치 크기
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_Y = Y_train[i:i+batch_size]
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    return model
