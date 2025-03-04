import os
import pickle
import hashlib
import time
from functools import wraps
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# 캐싱 디렉토리
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def generate_cache_key(func_name, args, kwargs):
    """
    함수 이름, 인자, 키워드 인자를 기반으로 캐시 키를 생성합니다.
    """
    # 인자들을 문자열로 변환
    args_str = str([str(arg) for arg in args])
    kwargs_str = str([(k, str(v)) for k, v in kwargs.items()])
    
    # 함수 이름과 인자들을 결합하여 해시 생성
    key_str = f"{func_name}_{args_str}_{kwargs_str}"
    return hashlib.md5(key_str.encode()).hexdigest()

def cache_result(expire_hours=24):
    """
    함수의 결과를 캐싱하는 데코레이터
    
    Args:
        expire_hours: 캐시 만료 시간 (시간 단위)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 캐시 키 생성
            cache_key = generate_cache_key(func.__name__, args, kwargs)
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
            
            # 캐시 파일 존재 및 만료 여부 확인
            if os.path.exists(cache_file):
                # 파일 수정 시간 확인
                mod_time = os.path.getmtime(cache_file)
                current_time = time.time()
                
                # 캐시가 만료되지 않았다면 캐시된 결과 반환
                if (current_time - mod_time) / 3600 < expire_hours:
                    with open(cache_file, 'rb') as f:
                        print(f"Cache hit: {func.__name__}")
                        return pickle.load(f)
            
            # 캐시가 없거나 만료된 경우 함수 실행
            result = func(*args, **kwargs)
            
            # 결과 캐싱
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            return result
        return wrapper
    return decorator

def clear_cache(func_name=None):
    """
    캐시를 삭제합니다.
    
    Args:
        func_name: 특정 함수의 캐시만 삭제하려면 함수 이름 지정
    """
    if func_name:
        for file in os.listdir(CACHE_DIR):
            if file.startswith(func_name):
                os.remove(os.path.join(CACHE_DIR, file))
    else:
        for file in os.listdir(CACHE_DIR):
            os.remove(os.path.join(CACHE_DIR, file))

def parallel_process(func, items, n_jobs=-1, backend='joblib', use_gpu=False, **kwargs):
    """
    함수를 병렬로 실행합니다.
    
    Args:
        func: 실행할 함수
        items: 함수에 전달할 항목 리스트
        n_jobs: 병렬 작업 수
        backend: 백엔드 ('joblib', 'torch', 'concurrent')
        use_gpu: GPU 사용 여부 (torch 백엔드에만 적용)
        **kwargs: 함수에 전달할 추가 인자
    
    Returns:
        결과 리스트
    """
    if backend == 'joblib':
        results = Parallel(n_jobs=n_jobs)(
            delayed(func)(item, **kwargs) for item in tqdm(items)
        )
    elif backend == 'torch':
        if use_gpu and torch.cuda.is_available():
            # GPU 병렬 처리
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                pool = mp.Pool(processes=min(num_gpus, len(items)))
                results = pool.map(func, items)
                pool.close()
                pool.join()
            else:
                # GPU가 없으면 CPU로 폴백
                pool = mp.Pool(processes=n_jobs if n_jobs > 0 else os.cpu_count())
                results = pool.map(func, items)
                pool.close()
                pool.join()
        else:
            # CPU 병렬 처리
            pool = mp.Pool(processes=n_jobs if n_jobs > 0 else os.cpu_count())
            results = pool.map(func, items)
            pool.close()
            pool.join()
    elif backend == 'concurrent':
        # concurrent.futures 백엔드
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            results = list(tqdm(executor.map(func, items), total=len(items)))
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    return results

def parallel_train_for_regions(train_func, regions, data, **kwargs):
    """
    여러 지역에 대해 병렬로 모델을 학습합니다.
    
    Args:
        train_func: 학습 함수
        regions: 지역 리스트
        data: 전체 데이터
        **kwargs: 학습 함수에 전달할 추가 인자
    
    Returns:
        지역별 모델 딕셔너리
    """
    def train_region_model(region):
        # 지역 데이터 필터링
        region_data = data[data['area'] == region].copy()
        if len(region_data) < kwargs.get('min_samples', 30):
            print(f"Insufficient data for region {region}. Skipping...")
            return region, None
        
        # 지역별 모델 학습
        model = train_func(region_data, **kwargs)
        return region, model
    
    # 병렬 처리로 각 지역 모델 학습
    n_jobs = kwargs.get('n_jobs', -1)
    backend = kwargs.get('backend', 'joblib')
    use_gpu = kwargs.get('use_gpu', False)
    
    results = parallel_process(
        train_region_model, regions, n_jobs=n_jobs, 
        backend=backend, use_gpu=use_gpu
    )
    
    # 결과를 딕셔너리로 변환
    region_models = {}
    for region, model in results:
        if model is not None:
            region_models[region] = model
    
    return region_models

def get_optimal_device_config():
    """
    최적의 디바이스 구성을 반환합니다.
    
    Returns:
        device: torch 디바이스
        use_mixed_precision: 혼합 정밀도 사용 여부
        num_workers: 데이터 로더 워커 수
    """
    # GPU 사용 가능 여부 확인 (CUDA 강제 활성화 시도)
    try:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            
        import torch.cuda
        torch.cuda.init()  # 명시적으로 CUDA 초기화
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            use_mixed_precision = True
            num_workers = min(4, os.cpu_count())
            
            # 사용 가능한 GPU 수 및 메모리 출력
            num_gpus = torch.cuda.device_count()
            print(f"사용 가능한 GPU: {num_gpus}개")
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"GPU {i}: {gpu_name}, 메모리: {gpu_memory:.2f} GB")
        else:
            # GPU 디버깅 정보 출력
            print("CUDA 가용성 검사 실패. 디버깅 정보:")
            print(f"torch.version.cuda: {torch.version.cuda}")
            print("NVIDIA-SMI 출력:")
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(result.stdout.decode('utf-8'))
            except:
                print("nvidia-smi를 실행할 수 없습니다.")
                
            device = torch.device("cpu")
            use_mixed_precision = False
            num_workers = min(4, os.cpu_count())
            print("GPU를 찾을 수 없음. CPU를 사용합니다.")
    except Exception as e:
        print(f"GPU 초기화 중 오류 발생: {str(e)}")
        device = torch.device("cpu")
        use_mixed_precision = False
        num_workers = min(4, os.cpu_count())
        print("오류로 인해 CPU를 사용합니다.")
    
    return device, use_mixed_precision, num_workers

def estimate_optimal_batch_size(model, input_shape, max_batch_size=1024, target_device=None):
    """
    메모리에 맞는 최적의 배치 크기를 추정합니다.
    
    Args:
        model: PyTorch 모델
        input_shape: 입력 텐서 형태 (seq_len, feature_dim)
        max_batch_size: 최대 배치 크기
        target_device: 목표 디바이스
    
    Returns:
        최적의 배치 크기
    """
    device = target_device if target_device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    
    model = model.to(device)
    
    # 시작 배치 크기
    batch_size = 4
    
    while batch_size <= max_batch_size:
        try:
            # 더미 입력 생성
            dummy_input = torch.randn(batch_size, *input_shape, device=device)
            
            # 순전파 및 역전파 테스트
            output = model(dummy_input)
            loss = output.sum()
            loss.backward()
            
            # 다음 배치 크기로 증가
            batch_size *= 2
            
            # 메모리 사용량이 많으면 일시 중지
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        except RuntimeError as e:
            # 메모리 부족 오류 발생 시
            print(f"배치 크기 {batch_size}에서 메모리 오류 발생")
            return batch_size // 2
    
    # 최대 배치 크기에 도달한 경우
    return max_batch_size

def memory_efficient_predict(model, X, batch_size=None, device=None):
    """
    메모리 효율적인 예측을 수행합니다.
    
    Args:
        model: PyTorch 모델
        X: 입력 데이터
        batch_size: 배치 크기 (None이면 자동 결정)
        device: torch 디바이스
    
    Returns:
        예측 결과
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 배치 크기가 지정되지 않은 경우 입력 크기를 기준으로 결정
    if batch_size is None:
        if X.shape[0] > 1000:
            batch_size = 256
        elif X.shape[0] > 100:
            batch_size = 64
        else:
            batch_size = 32
    
    model = model.to(device)
    model.eval()
    
    # 입력이 텐서가 아니면 변환
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    
    preds = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size].to(device)
            batch_preds = model(batch_X)
            preds.append(batch_preds.cpu().numpy())
    
    return np.vstack(preds)

def clean_gpu_memory():
    """GPU 메모리 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
def move_tensors_to_device(tensors, device):
    """텐서를 지정된 디바이스로 이동"""
    if isinstance(tensors, dict):
        return {k: move_tensors_to_device(v, device) for k, v in tensors.items()}
    elif isinstance(tensors, list):
        return [move_tensors_to_device(t, device) for t in tensors]
    elif isinstance(tensors, tuple):
        return tuple(move_tensors_to_device(list(tensors), device))
    elif isinstance(tensors, torch.Tensor):
        return tensors.to(device)
    else:
        return tensors
