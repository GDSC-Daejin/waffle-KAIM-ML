import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, X_train, Y_train, epochs=100, batch_size=32):
    """
    [원리 설명]
    - 주어진 학습 데이터(X_train)와 타겟 데이터(Y_train)를 사용하여 LSTM 모델을 학습시킵니다.
    - 손실 함수로는 MSELoss(평균제곱오차)를 사용하고, Adam 옵티마이저를 통해 가중치를 업데이트합니다.
    - 에포크와 배치 단위로 학습을 진행하며, 일정 간격마다 손실 값을 출력하여 학습 진행 상황을 모니터링합니다.
    
    반환:
      - 학습이 완료된 모델
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
