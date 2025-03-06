import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from model import create_model

def train_model(model, X_train, Y_train, epochs=50, batch_size=32, lr=0.001, verbose=0):
    """
    모델을 훈련하는 함수입니다.
    
    Parameters:
        - model: 훈련할 모델 (예: LSTMModel)
        - X_train: 훈련 입력 데이터 (NumPy 배열 또는 변환 가능한 형식)
        - Y_train: 훈련 목표 데이터 (NumPy 배열 또는 변환 가능한 형식)
        - epochs: 훈련할 에포크 수
        - batch_size: 배치 크기
        - lr: 학습률
        - verbose: 학습 진행 출력 상세도
    Returns:
        - 학습된 모델과 학습 히스토리
    """
    # 안전하게 데이터 변환 - 리스트 형태의 요소 처리
    def safe_convert_to_array(data):
        if isinstance(data, np.ndarray):
            return data
            
        if isinstance(data, torch.Tensor):
            return data.numpy()
            
        if hasattr(data, 'values'):  # pandas DataFrame 또는 Series
            return data.values.astype(np.float64)
        
        # 데이터가 리스트인 경우, 각 요소에 대해 처리
        if isinstance(data, list):
            # 모든 요소가 숫자인지 확인
            is_numeric = all(isinstance(x, (int, float)) for x in data)
            if is_numeric:
                return np.array(data, dtype=np.float64)
            else:
                # 리스트의 리스트인 경우 - 첫 번째 요소만 사용
                cleaned_data = []
                for item in data:
                    if isinstance(item, list):
                        cleaned_data.append(item[0] if item else 0.0)
                    else:
                        cleaned_data.append(float(item) if item is not None else 0.0)
                return np.array(cleaned_data, dtype=np.float64)
        
        # 기타 경우
        try:
            return np.array(data, dtype=np.float64)
        except:
            raise ValueError("데이터를 numpy 배열로 변환할 수 없습니다.")
    
    try:
        # 원본 형태 기록
        original_shape = None
        if hasattr(X_train, 'shape'):
            original_shape = X_train.shape
            print(f"원본 X_train 형태: {original_shape}")
        elif isinstance(X_train, torch.Tensor):
            original_shape = X_train.shape
            print(f"원본 X_train 형태 (텐서): {original_shape}")
        
        # X_train과 Y_train을 안전하게 numpy 배열로 변환
        X_train_np = safe_convert_to_array(X_train)
        Y_train_np = safe_convert_to_array(Y_train)
        
        # 형태 검사 및 로깅
        print(f"변환 후 X_train 형태: {X_train_np.shape}, Y_train 형태: {Y_train_np.shape}")
        
        # 데이터 유효성 확인
        if np.isnan(X_train_np).any() or np.isinf(X_train_np).any():
            print("경고: X_train에 NaN 또는 Inf 값이 있습니다. 0으로 대체합니다.")
            X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.isnan(Y_train_np).any() or np.isinf(Y_train_np).any():
            print("경고: Y_train에 NaN 또는 Inf 값이 있습니다. 0으로 대체합니다.")
            Y_train_np = np.nan_to_num(Y_train_np, nan=0.0, posinf=0.0, neginf=0.0)
        
        # PyTorch 텐서로 변환
        X_train_tensor = torch.FloatTensor(X_train_np)
        Y_train_tensor = torch.FloatTensor(Y_train_np)
        
        print(f"텐서 변환 후 X_train 형태: {X_train_tensor.shape}, Y_train 형태: {Y_train_tensor.shape}")
        
    except Exception as e:
        print(f"데이터 변환 중 오류: {str(e)}")
        # 대체 로직: numpy 배열로 강제 변환 시도
        try:
            # 원본 형태 확인
            if hasattr(X_train, 'shape') and len(X_train.shape) == 3:
                batch_size, seq_len, feat_dim = X_train.shape
                print(f"3D 데이터 감지: batch_size={batch_size}, seq_len={seq_len}, feat_dim={feat_dim}")
                
                # 3D 형태를 유지하며 변환
                X_train_np = np.zeros((batch_size, seq_len, feat_dim), dtype=np.float64)
                for i in range(batch_size):
                    for j in range(seq_len):
                        for k in range(feat_dim):
                            try:
                                val = X_train[i, j, k]
                                if isinstance(val, list):
                                    val = val[0] if val else 0.0
                                X_train_np[i, j, k] = float(val)
                            except:
                                X_train_np[i, j, k] = 0.0
                
                # Y_train 변환
                Y_train_np = np.zeros((batch_size, Y_train.shape[1]), dtype=np.float64)
                for i in range(batch_size):
                    for j in range(Y_train.shape[1]):
                        try:
                            val = Y_train[i, j]
                            if isinstance(val, list):
                                val = val[0] if val else 0.0
                            Y_train_np[i, j] = float(val)
                        except:
                            Y_train_np[i, j] = 0.0
            else:
                # 차원을 알 수 없는 경우 - 기존 대체 로직 사용
                X_train_flat = np.array([float(x) if not isinstance(x, list) else float(x[0] if x else 0) 
                                     for x in np.array(X_train).flatten()], dtype=np.float64)
                Y_train_flat = np.array([float(y) if not isinstance(y, list) else float(y[0] if y else 0) 
                                     for y in np.array(Y_train).flatten()], dtype=np.float64)
                
                # 3차원 구조로 복원 (LSTM용)
                if isinstance(X_train, torch.Tensor) and len(X_train.shape) == 3:
                    batch_size, seq_len, feat_dim = X_train.shape
                    X_train_np = X_train_flat.reshape(batch_size, seq_len, feat_dim)
                else:
                    # 기본 가정: 배치 크기는 첫 번째 차원, 나머지는 특성으로 처리
                    X_train_np = X_train_flat.reshape(len(X_train), 1, -1)
                
                # Y_train은 2차원으로 처리
                if isinstance(Y_train, torch.Tensor):
                    Y_train_np = Y_train_flat.reshape(Y_train.shape)
                else:
                    Y_train_np = Y_train_flat.reshape(len(Y_train), -1)
            
            X_train_tensor = torch.FloatTensor(X_train_np)
            Y_train_tensor = torch.FloatTensor(Y_train_np)
            
            print(f"대체 변환 사용 - X_train 형태: {X_train_tensor.shape}, Y_train 형태: {Y_train_tensor.shape}")
        except Exception as error:
            print(f"대체 변환도 실패: {str(error)}")
            raise ValueError("데이터를 텐서로 변환할 수 없습니다. 데이터 형식을 확인하세요.")
    
    # LSTM에 적합한 형태인지 확인
    if len(X_train_tensor.shape) == 2:  # 2차원인 경우 3차원으로 변환 (LSTM 요구사항)
        print("경고: 입력이 2차원입니다. 3차원으로 변환합니다.")
        X_train_tensor = X_train_tensor.unsqueeze(0)  # (seq_len, features) -> (1, seq_len, features)
    
    # 모델 디바이스 확인
    device = next(model.parameters()).device
    
    # 데이터로더 준비
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size if X_train_tensor.shape[0] > 1 else 1,  # 단일 샘플인 경우 배치 크기 조정
        shuffle=True
    )

    # 옵티마이저와 손실 함수 설정
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # 훈련 기록
    history = {"loss": []}
    
    # 훈련 루프
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 입력 형태 확인 (첫 번째 배치만)
            if epoch == 0 and batch_idx == 0:
                print(f"첫 번째 배치 입력 형태: {inputs.shape}")
                # 배치 차원이 없는 경우 추가
                if len(inputs.shape) == 2:
                    inputs = inputs.unsqueeze(0)
                    print(f"배치 차원 추가 후 형태: {inputs.shape}")
            
            # 중요: 입력과 타겟을 모델과 같은 디바이스로 이동
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        history["loss"].append(avg_loss)
        
        if verbose > 0 and (epoch + 1) % verbose == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

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
