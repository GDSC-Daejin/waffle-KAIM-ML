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

def train_model(model, X_train, Y_train, X_val=None, Y_val=None, 
               epochs=100, batch_size=64, learning_rate=0.001,
               patience=10, verbose=1, device=None):
    """
    모델 학습 함수
    
    Args:
        model: 학습할 PyTorch 모델
        X_train, Y_train: 학습 데이터
        X_val, Y_val: 검증 데이터 (없으면 학습 데이터의 일부를 검증에 사용)
        epochs: 학습 에포크 수
        batch_size: 배치 크기 (성능 최적화를 위해 증가)
        learning_rate: 학습률
        patience: 조기 종료 인내심
        verbose: 출력 상세도
        device: 학습에 사용할 디바이스
        
    Returns:
        학습된 모델, 손실 기록
    """
    # 디바이스 설정
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    # 혼합 정밀도 연산 설정 (GPU 효율성 향상)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # 옵티마이저 및 손실 함수 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    # 검증 데이터가 없으면 학습 데이터 분할
    if X_val is None or Y_val is None:
        val_size = int(0.2 * len(X_train))
        X_val = X_train[-val_size:]
        Y_val = Y_train[-val_size:]
        X_train = X_train[:-val_size]
        Y_train = Y_train[:-val_size]
    
    # 학습 데이터를 TensorDataset으로 변환
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    
    # DataLoader 설정 (병렬 처리 향상)
    num_workers = min(8, os.cpu_count()) if device.type == 'cuda' else 0
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size*2, shuffle=False,
        num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False
    )
    
    # 조기 종료를 위한 변수
    best_val_loss = float('inf')
    no_improve_epochs = 0
    best_model_state = None
    train_losses, val_losses = [], []
    
    # 학습 시작
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # tqdm으로 진행률 표시
        train_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{epochs}]') if verbose > 0 else train_loader
        
        for inputs, targets in train_bar:
            # 데이터를 해당 디바이스로 이동
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            if scaler is not None:  # 혼합 정밀도 사용
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # 스케일링된 역전파
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            if verbose > 0:
                train_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        # 에포크 종료 후 평균 손실 계산
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 검증
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # 진행 상황 출력
        if verbose > 0 and (epoch+1) % (epochs//10 or 1) == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 최고 모델 저장 및 조기 종료 확인
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            
        # 조기 종료
        if no_improve_epochs >= patience:
            if verbose > 0:
                print(f"조기 종료: {epoch+1} 에폭에서 검증 손실 개선 없음")
            break
    
    # 최고 모델 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses}

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
