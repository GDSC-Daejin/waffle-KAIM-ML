import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np  # np를 추가

from model import create_model

def train_model(model, X_train, Y_train, epochs=50, batch_size=32, lr=0.001):
    """
    모델을 훈련하는 함수입니다.
    
    Parameters:
        - model: 훈련할 모델 (예: LSTMModel)
        - X_train: 훈련 입력 데이터 (NumPy 배열)
        - Y_train: 훈련 목표 데이터 (NumPy 배열)
        - epochs: 훈련할 에포크 수
        - batch_size: 배치 크기
        - lr: 학습률
    Returns:
        - 학습된 모델
    """
    X_train = np.array(X_train, dtype=np.float64)  # np.array()로 변환
    Y_train = np.array(Y_train, dtype=np.float64)  # np.array()로 변환

    X_train_tensor = torch.FloatTensor(X_train)  # np.float64로 변환
    Y_train_tensor = torch.FloatTensor(Y_train)  # np.float64로 변환
    
    # 데이터로더 준비
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 옵티마이저와 손실 함수 설정
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # 훈련 루프
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader)}")

    return model

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
        trained_model = train_model(model, X_train, Y_train, epochs=epochs, batch_size=batch_size, lr=lr)
        
        ensemble_models.append(trained_model)
    
    return ensemble_models
