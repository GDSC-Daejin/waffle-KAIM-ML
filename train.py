import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import concurrent.futures
from joblib import Parallel, delayed
import time
from tqdm import tqdm
import os
from sklearn.model_selection import TimeSeriesSplit

def train_model(model, X_train, Y_train, epochs=100, batch_size=32, learning_rate=0.001, 
                validation_data=None, patience=10, verbose=1):
    """
    단일 모델 학습 함수
    
    Args:
        model: 학습시킬 모델
        X_train: 학습 입력 데이터
        Y_train: 학습 타겟 데이터
        epochs: 학습 에폭 수
        batch_size: 배치 크기
        learning_rate: 학습률
        validation_data: (X_val, Y_val) 형태의 검증 데이터
        patience: 조기 종료를 위한 인내심 횟수
        verbose: 출력 상세도
        
    Returns:
        학습된 모델과 학습 기록
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience//2, factor=0.5)
    
    X_train = torch.tensor(X_train.numpy() if isinstance(X_train, torch.Tensor) else X_train, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_train.numpy() if isinstance(Y_train, torch.Tensor) else Y_train, dtype=torch.float32).to(device)
    
    if validation_data:
        X_val, Y_val = validation_data
        X_val = torch.tensor(X_val.numpy() if isinstance(X_val, torch.Tensor) else X_val, dtype=torch.float32).to(device)
        Y_val = torch.tensor(Y_val.numpy() if isinstance(Y_val, torch.Tensor) else Y_val, dtype=torch.float32).to(device)
    
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        # 미니배치 학습
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_Y = Y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = total_loss / max(1, batch_count)
        history['train_loss'].append(avg_train_loss)
        
        # 검증
        if validation_data:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, Y_val).item()
                history['val_loss'].append(val_loss)
                
                # 학습률 스케줄러 업데이트
                scheduler.step(val_loss)
                
                # 조기 종료
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
        
        if verbose > 0 and (epoch+1) % max(1, epochs//10) == 0:
            val_msg = f", Val Loss: {val_loss:.4f}" if validation_data else ""
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}{val_msg}")
    
    # 조기 종료된 경우 최적 모델 복원
    if best_model_state and validation_data:
        model.load_state_dict(best_model_state)
    
    return model, history

def train_model_with_cv(model_creator, X, Y, cv=5, epochs=100, batch_size=32, **train_kwargs):
    """
    시계열 교차 검증을 사용하여 모델을 학습하고 평가
    
    Args:
        model_creator: 모델 생성 함수
        X: 전체 입력 데이터
        Y: 전체 타겟 데이터
        cv: 교차 검증 폴드 수
        epochs: 각 폴드 학습의 에폭 수
        
    Returns:
        최종 모델과 각 폴드의 검증 점수
    """
    tscv = TimeSeriesSplit(n_splits=cv)
    val_scores = []
    best_val_score = float('inf')
    best_model = None
    
    for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        Y_train_fold, Y_val_fold = Y[train_idx], Y[val_idx]
        
        # 각 폴드마다 새로운 모델 생성
        model = model_creator()
        
        model, _ = train_model(
            model, 
            X_train_fold, 
            Y_train_fold, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(X_val_fold, Y_val_fold),
            **train_kwargs
        )
        
        # 검증 성능 평가
        device = next(model.parameters()).device
        X_val_fold = torch.tensor(X_val_fold if not isinstance(X_val_fold, torch.Tensor) else X_val_fold.numpy(), dtype=torch.float32).to(device)
        Y_val_fold = torch.tensor(Y_val_fold if not isinstance(Y_val_fold, torch.Tensor) else Y_val_fold.numpy(), dtype=torch.float32).to(device)
        
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_fold)
            val_score = nn.MSELoss()(val_preds, Y_val_fold).item()
        
        val_scores.append(val_score)
        
        # 최고 성능 모델 저장
        if val_score < best_val_score:
            best_val_score = val_score
            best_model = model
    
    return best_model, val_scores

def train_ensemble_models(model_configs, X_train, Y_train, ensemble_size=5, cpu_models=None, 
                          use_gpu=True, n_jobs=-1, **train_kwargs):
    """
    앙상블을 위한 여러 모델을 병렬로 학습
    
    Args:
        model_configs: 각 모델 구성의 리스트 (모델 타입, 하이퍼파라미터 등)
        X_train: 학습 입력 데이터
        Y_train: 학습 타겟 데이터
        ensemble_size: 각 모델 구성에서 학습할 모델 수
        cpu_models: CPU에서 학습할 모델 타입 리스트 (예: ['transformer'])
        use_gpu: GPU 사용 여부
        n_jobs: 병렬 처리 작업 수 (CPU용)
        
    Returns:
        학습된 모델들과 모델별 가중치
    """
    if cpu_models is None:
        cpu_models = ['transformer']  # 기본적으로 transformer는 CPU에서 훈련
    
    # GPU 사용 가능한지 확인
    gpu_available = torch.cuda.is_available() and use_gpu
    
    # CPU와 GPU 모델 분리
    gpu_model_configs = []
    cpu_model_configs = []
    
    for config in model_configs:
        model_type = config["model_type"]
        if model_type in cpu_models or not gpu_available:
            cpu_model_configs.append(config)
        else:
            gpu_model_configs.append(config)
    
    all_models = []
    start_time = time.time()
    
    # GPU 모델 학습
    if gpu_model_configs and gpu_available:
        print("Starting GPU model training...")
        gpu_models = []
        
        for config in gpu_model_configs:
            for i in range(ensemble_size):
                config_name = f"{config['model_type']}_{i+1}"
                print(f"Training {config_name} on GPU...")
                
                from model import create_model
                model = create_model(**config)
                
                # 검증 데이터 분할
                val_size = int(len(X_train) * 0.2)
                X_train_sub, X_val = X_train[:-val_size], X_train[-val_size:]
                Y_train_sub, Y_val = Y_train[:-val_size], Y_train[-val_size:]
                
                trained_model, _ = train_model(
                    model, 
                    X_train_sub, 
                    Y_train_sub, 
                    validation_data=(X_val, Y_val),
                    **train_kwargs
                )
                
                gpu_models.append(trained_model.cpu())  # CPU로 이동하여 메모리 확보
        
        all_models.extend(gpu_models)
    
    # CPU 모델 병렬 학습
    if cpu_model_configs:
        print("Starting CPU model training in parallel...")
        
        def train_one_model(config, model_idx):
            config_name = f"{config['model_type']}_{model_idx+1}"
            print(f"Training {config_name} on CPU...")
            
            from model import create_model
            model = create_model(**config)
            
            # 검증 데이터 분할
            val_size = int(len(X_train) * 0.2)
            X_train_sub = X_train[:-val_size].numpy() if isinstance(X_train, torch.Tensor) else X_train[:-val_size]
            Y_train_sub = Y_train[:-val_size].numpy() if isinstance(Y_train, torch.Tensor) else Y_train[:-val_size]
            X_val = X_train[-val_size:].numpy() if isinstance(X_train, torch.Tensor) else X_train[-val_size:]
            Y_val = Y_train[-val_size:].numpy() if isinstance(Y_train, torch.Tensor) else Y_train[-val_size:]
            
            trained_model, _ = train_model(
                model, 
                X_train_sub, 
                Y_train_sub, 
                validation_data=(X_val, Y_val),
                **train_kwargs
            )
            
            return trained_model
        
        cpu_jobs = []
        for config in cpu_model_configs:
            for i in range(ensemble_size):
                cpu_jobs.append((config, i))
        
        n_jobs = n_jobs if n_jobs != -1 else os.cpu_count()
        n_jobs = min(n_jobs, len(cpu_jobs))
        
        cpu_models = Parallel(n_jobs=n_jobs)(
            delayed(train_one_model)(config, idx) for config, idx in cpu_jobs
        )
        
        all_models.extend(cpu_models)
    
    print(f"All models trained in {time.time() - start_time:.2f} seconds")
    
    # 모델별 가중치 계산 (여기서는 동일 가중치 사용)
    weights = np.ones(len(all_models)) / len(all_models)
    
    return all_models, weights
