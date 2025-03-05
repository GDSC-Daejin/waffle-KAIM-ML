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

def optimize_gpu_tensor_ops(batch_size=None, mixed_precision=True):
    """
    GPU 텐서 연산 최적화
    
    Args:
        batch_size: 배치 크기 (None이면 자동 계산)
        mixed_precision: 혼합 정밀도 사용 여부
    
    Returns:
        최적화된 배치 크기
    """
    if not torch.cuda.is_available():
        return 32  # CPU 기본값
    
    try:
        # CUDA 컨텍스트 초기화 확인
        dummy_tensor = torch.zeros(1, device='cuda')
        del dummy_tensor
    except RuntimeError as e:
        print(f"CUDA 초기화 오류 발생: {e}")
        print("CPU 모드로 전환합니다.")
        return 32  # CPU 기본값
    
    # GPU 메모리 크기에 따른 최적의 배치 크기 계산
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    if batch_size is None:
        # GPU 메모리에 최적화된 배치 크기 계산
        if mixed_precision:
            # FP16/FP32 혼합 정밀도 사용 시
            batch_size = int(min(512, 32 * (gpu_mem_gb / 4)))
        else:
            # FP32 정밀도만 사용 시
            batch_size = int(min(256, 16 * (gpu_mem_gb / 4)))
    
    # 메모리 프래그멘테이션 방지
    torch.cuda.empty_cache()
    
    # NVIDIA 최적화된 커널 사용
    torch.backends.cudnn.benchmark = True
    
    print(f"GPU 메모리: {gpu_mem_gb:.2f}GB, 최적화된 배치 크기: {batch_size}")
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

def setup_cuda_multiprocessing():
    """
    CUDA와 멀티프로세싱 환경을 설정하여 호환성 문제 해결
    
    이 함수는 CUDA와 Python 멀티프로세싱을 함께 사용할 때 발생하는
    초기화 문제를 해결하는 데 도움이 됩니다.
    """
    # 멀티프로세싱 시작 방법을 'spawn'으로 설정 (가장 안전한 방법)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # 이미 설정되어 있을 경우 무시
        pass
    
    # CUDA 초기화를 강제로 수행
    if torch.cuda.is_available():
        # 모든 CUDA 장치 초기화
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.tensor([0], device=f'cuda:{i}')  # 각 장치에서 간단한 텐서 생성
        
        # 메모리 캐시 비우기
        torch.cuda.empty_cache()
        
        print("CUDA 멀티프로세싱 환경이 설정되었습니다.")
        return True
    return False

def get_optimal_worker_count(use_gpu=True):
    """
    시스템에 최적화된 DataLoader 워커 수 결정
    
    Args:
        use_gpu: GPU를 사용할지 여부
        
    Returns:
        최적의 워커 수
    """
    if not use_gpu or not torch.cuda.is_available():
        # CPU 모드에서는 코어의 75% 정도 사용 (최소 1개)
        return max(1, int(os.cpu_count() * 0.75)) if os.cpu_count() else 2
    else:
        # GPU 모드에서는 워커 수를 제한하여 CUDA 초기화 문제 방지
        # GPU 별 워커 수 제한
        workers_per_gpu = 2
        num_gpus = torch.cuda.device_count()
        
        # CPU 코어 수와 GPU 기반 권장 워커 수 중 작은 값 선택
        recommended = min(workers_per_gpu * num_gpus, os.cpu_count() or 4)
        
        # 최소 워커 수는 0 (메인 프로세스에서만 로딩)
        return max(0, recommended - 1)  # 메인 프로세스를 위해 1 감소
