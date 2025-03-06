import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from model import create_model

def train_model(model, X_train, Y_train, epochs=50, batch_size=32, lr=0.001, patience=30, min_delta=0.0001, verbose=0):
    """모델을 훈련하는 함수"""
    # 디바이스 확인
    device = next(model.parameters()).device
    
    # X_train, Y_train을 모델과 같은 디바이스로 이동
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    
    # 학습 준비
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    # 히스토리 저장
    history = {'loss': []}
    
    # 조기 종료 설정 - 매개변수 값 사용
    best_loss = float('inf')
    patience_counter = 0
    
    # 학습률 스케줄러 추가 - verbose 파라미터 제거
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # 학습 배치 설정
    batch_size = min(batch_size, len(X_train))
    
    # 에포크별 학습
    for epoch in range(epochs):
        # 그라디언트 초기화
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        # 무작위 인덱스
        indices = torch.randperm(X_train.size(0))
        
        for start_idx in range(0, X_train.size(0), batch_size):
            try:
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
                
                # 그라디언트 클리핑 (폭주 방지)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 파라미터 업데이트
                optimizer.step()
                
                # 손실 누적
                running_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"배치 처리 중 오류 발생: {e}, 이 배치는 건너뜁니다.")
                continue
        
        # 에포크 평균 손실
        epoch_loss = running_loss / max(1, num_batches)
        history['loss'].append(epoch_loss)
        
        # 진행 상황 출력
        if verbose > 0 and (epoch + 1) % verbose == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
        
        # 조기 종료 조건 개선 - min_delta 사용
        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            patience_counter = 0
            # 최적 모델 저장
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"조기 종료! {patience}번의 에포크 동안 {min_delta} 이상의 개선이 없었습니다.")
                # 최적 상태로 복원
                model.load_state_dict(best_model_state)
                break
        
        # 학습률 스케줄러 업데이트
        scheduler.step(epoch_loss)
    
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
