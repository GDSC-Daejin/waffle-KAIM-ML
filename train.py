import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from model import create_model

def train_model(model, X_train, Y_train, epochs=50, batch_size=32, lr=0.001, verbose=0):
    """모델을 훈련하는 함수"""
    
    # 디바이스 확인 - 모델과 데이터가 같은 디바이스에 있어야 함
    device = next(model.parameters()).device
    
    # X_train, Y_train을 모델과 같은 디바이스로 이동
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    
    # 학습 준비
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    # 훈련 히스토리 저장
    history = {'loss': []}
    
    # 모델을 훈련 모드로 설정
    model.train()
    
    # 학습 배치 설정 (배치 크기가 데이터셋보다 크면 조정)
    batch_size = min(batch_size, len(X_train))
    
    # 에포크별 학습
    for epoch in range(epochs):
        # 배치 인덱스 생성 (무작위)
        indices = torch.randperm(X_train.size(0))
        
        # 배치 단위로 훈련
        total_loss = 0
        num_batches = 0
        
        for start_idx in range(0, X_train.size(0), batch_size):
            # 배치 인덱스
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            # 배치 데이터
            X_batch = X_train[batch_indices]
            Y_batch = Y_train[batch_indices]
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(X_batch)
            
            # 손실 계산
            loss = criterion(outputs, Y_batch)
            
            # 역전파
            loss.backward()
            
            # 파라미터 업데이트
            optimizer.step()
            
            # 손실 누적
            total_loss += loss.item()
            num_batches += 1
        
        # 에포크 평균 손실 계산
        avg_loss = total_loss / num_batches
        history['loss'].append(avg_loss)
        
        # 학습 과정 출력
        if verbose > 0 and (epoch + 1) % verbose == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return model, history

def train_ensemble_models(model_configs, X_train, Y_train, ensemble_size=3, epochs=50, batch_size=32, lr=0.001, use_gpu=True):
    """
    여러 모델을 앙상블로 훈련하는 함수입니다.
    
    Parameters:
        - model_configs: 모델 설정을 포함하는 딕셔너리 리스트
        - X_train: 훈련 입력 데이터 (NumPy 배열)
        - Y_train: 훈련 목표 데이터 (NumPy 배열)
        - ensemble_size: 앙상블에 사용할 모델 수
        - epochs: 훈련할 에포크 수
        - batch_size: 배치 크기
        - lr: 학습률
        - use_gpu: GPU 사용 여부
    Returns:
        - 학습된 앙상블 모델 리스트
    """
    ensemble_models = []
    
    for config in model_configs:
        model_type = config.get('model_type', 'lstm')
        input_dim = config.get('input_dim', X_train.shape[1])
        hidden_dim = config.get('hidden_dim', 64)
        output_dim = config.get('output_dim', Y_train.shape[1])
        
        model = create_model(model_type, input_dim, hidden_dim, output_dim, **config)
        
        # GPU 사용 여부에 따라 모델을 이동
        if use_gpu and torch.cuda.is_available():
            model = model.cuda()
        
        print(f"훈련 중: {model_type}")
        
        # 앙상블 모델 훈련
        trained_model, _ = train_model(model, X_train, Y_train, epochs=epochs, batch_size=batch_size, lr=lr)
        
        ensemble_models.append(trained_model)
    
    return ensemble_models
