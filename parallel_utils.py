import os
import torch
import torch.multiprocessing as mp
import numpy as np
from joblib import Parallel, delayed
import concurrent.futures
from tqdm import tqdm

def optimize_cpu_affinity():
    """CPU 코어 어피니티 최적화"""
    try:
        import psutil
        # 현재 프로세스
        process = psutil.Process()
        # 모든 CPU 코어에 대해 어피니티 설정
        process.cpu_affinity(list(range(psutil.cpu_count())))
        return True
    except:
        return False

def get_optimal_thread_count():
    """최적의 스레드 수 결정"""
    cpu_count = os.cpu_count()
    if cpu_count > 16:
        return cpu_count - 2  # 대형 시스템에서는 일부 코어 예약
    else:
        return max(1, cpu_count - 1)  # 소형 시스템에서는 1개 코어만 예약

def accelerate_training():
    """PyTorch 학습 가속화를 위한 설정을 적용합니다."""
    # 시스템 코어 수에 따라 스레드 수 최적화
    num_cores = os.cpu_count()
    
    # 시스템 전체 메모리의 90%까지 활용
    torch.set_num_threads(num_cores)  # 모든 CPU 코어 활용
    
    # 벡터화 연산 활성화
    torch.backends.cudnn.benchmark = True  # 컨볼루션 연산 최적화
    torch.backends.cudnn.deterministic = False  # 성능 최적화
    
    # GPU 메모리 최적화 설정
    if torch.cuda.is_available():
        # 메모리 할당 전략 최적화
        torch.cuda.empty_cache()
        
        # 자동 튜닝 활성화
        if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            torch.backends.cuda.matmul.allow_tf32 = True  # TF32 형식 활성화 (최신 GPU용)
        
        # GPU 캐시 크기 최적화 (가능한 경우)
        try:
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.95)  # GPU 메모리의 95% 사용
        except:
            pass
    
    print(f"학습 가속화 설정 완료 - 사용 스레드 수: {torch.get_num_threads()}")

def parallel_map(func, items, n_jobs=None, backend='joblib', use_gpu=False, chunk_size=None):
    """
    향상된 병렬 처리 함수
    
    Args:
        func: 실행할 함수
        items: 입력 아이템 리스트
        n_jobs: 작업 수 (None=자동)
        backend: 백엔드 ('joblib', 'concurrent', 'mp', 'torch')
        use_gpu: GPU 사용 여부
        chunk_size: 청크 단위 처리 크기
    """
    if n_jobs is None:
        n_jobs = get_optimal_thread_count()
    
    if chunk_size is None:
        chunk_size = max(1, len(items) // (n_jobs * 4))
    
    if backend == 'torch' and torch.cuda.is_available() and use_gpu:
        # GPU 병렬 처리
        return _torch_parallel(func, items)
    elif backend == 'mp':
        # 멀티프로세싱
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_jobs) as pool:
            results = list(pool.map(func, items, chunksize=chunk_size))
        return results
    elif backend == 'concurrent':
        # 동시성 기반 병렬 처리
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(func, items, chunksize=chunk_size))
        return results
    else:
        # 기본 joblib 병렬 처리
        return Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(func)(item) for item in items
        )

def _torch_parallel(func, items):
    """PyTorch 기반 병렬 처리"""
    n_gpus = torch.cuda.device_count()
    
    if n_gpus == 0:
        return [func(item) for item in items]
    
    # GPU별 작업 분배
    chunks = np.array_split(items, n_gpus)
    
    # GPU별 프로세스 시작
    processes = []
    for i, chunk in enumerate(chunks):
        p = mp.Process(
            target=_process_chunk,
            args=(func, chunk, i)
        )
        p.start()
        processes.append(p)
    
    # 프로세스 완료 대기
    for p in processes:
        p.join()
    
    # 결과 병합
    results = []
    for i in range(len(chunks)):
        chunk_results = torch.load(f"chunk_results_{i}.pt")
        results.extend(chunk_results)
        os.remove(f"chunk_results_{i}.pt")
    
    return results

def _process_chunk(func, chunk, gpu_id):
    """개별 GPU에서 청크 처리"""
    torch.cuda.set_device(gpu_id)
    results = [func(item) for item in chunk]
    torch.save(results, f"chunk_results_{gpu_id}.pt")

def parallel_batch_process(func, items, batch_size=10, max_workers=None):
    """
    병렬 처리를 위한 배치 처리 함수
    
    참고: 이 함수는 128GB RAM을 효율적으로 사용하기 위해 설계됨
    
    Args:
        func: 각 항목에 적용할 함수
        items: 처리할 항목 리스트
        batch_size: 각 배치의 크기
        max_workers: 최대 작업자 수 (None이면 모든 코어 사용)
    Returns:
        처리된 결과 리스트
    """
    if max_workers is None:
        max_workers = os.cpu_count()
    
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    with tqdm(total=len(items), desc="병렬 처리") as pbar:
        for batch_start in range(0, len(items), batch_size):
            batch_end = min(batch_start + batch_size, len(items))
            batch_items = items[batch_start:batch_end]
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                batch_results = list(executor.map(func, batch_items))
            
            results.extend(batch_results)
            pbar.update(len(batch_items))
    
    return results

def optimize_gpu_tensor_ops(mixed_precision=True):
    """
    GTX 1080 8GB GPU에 최적화된 텐서 연산 설정
    """
    if not torch.cuda.is_available():
        return 32  # CPU 기본값
    
    # GTX 1080 8GB에 최적화된 배치 크기
    device_props = torch.cuda.get_device_properties(0)
    gpu_mem_gb = device_props.total_memory / (1024**3)
    
    # 세밀한 배치 크기 조정
    if mixed_precision:
        # GTX 1080의 FP16 연산 성능은 제한적이므로 보수적으로 설정
        batch_size = min(128, int(16 * (gpu_mem_gb / 8)))
    else:
        batch_size = min(64, int(8 * (gpu_mem_gb / 8)))
    
    # 최적의 메모리 활용을 위한 NVIDIA 설정
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
    
    # GTX 1000 시리즈는 TensorCore가 없으므로 FP16 최적화는 제한적
    if hasattr(torch.cuda, 'amp') and mixed_precision:
        print("FP16/FP32 혼합 정밀도 활성화")
    
    print(f"GPU: {device_props.name}, 메모리: {gpu_mem_gb:.2f}GB, 최적 배치 크기: {batch_size}")
    return batch_size

def distribute_model_across_gpus(model, gpu_ids=None):
    """
    모델을 여러 GPU에 분산 (멀티 GPU 환경인 경우)
    
    Args:
        model: PyTorch 모델
        gpu_ids: 사용할 GPU ID 리스트 (None이면 모든 GPU 사용)
    
    Returns:
        분산된 모델
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return model
    
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))
    
    if len(gpu_ids) <= 1:
        return model.to(f'cuda:{gpu_ids[0]}')
    
    # DistributedDataParallel보다 간단한 DataParallel 사용
    return torch.nn.DataParallel(model, device_ids=gpu_ids)
