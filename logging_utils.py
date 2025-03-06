import logging
import time
import datetime
import pandas as pd
import os
import torch
import psutil
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
from tqdm import tqdm
from datetime import datetime

class PerformanceLogger:
    """
    연산 성능 및 리소스 사용량을 로깅하는 클래스
    """
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 로깅 설정
        self.logger = logging.getLogger("PerformanceLogger")
        self.logger.setLevel(logging.INFO)
        # 로그 파일 핸들러
        # 로그 파일 핸들러
        log_file = os.path.join(log_dir, f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 통계 저장
        self.stats = {
            'train_time': [],
            'train_batch_time': [],
            'eval_time': [],
            'predict_time': [],
            'gpu_usage': [],
            'cpu_usage': [],
            'memory_usage': [],
            'batch_sizes': [],
            'model_params': {},
        }
        
        # 시작 시간
        self.start_time = time.time()
    
    def log_hardware_info(self):
        """시스템 하드웨어 정보 로깅"""
        # CPU 정보
        cpu_info = f"CPU 코어 수: {cpu_count}"
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        memory_info = f"총 메모리: {memory.total / (1024**3):.2f}GB, 사용 가능: {memory.available / (1024**3):.2f}GB"
        
        # GPU 정보
        gpu_info = "GPU: "
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_info += f"{gpu_count}개 발견\n"
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpu_info += f"  GPU {i}: {gpu_name}, 메모리: {gpu_memory:.2f}GB\n"
        else:
            gpu_info += "사용 불가"
        
        self.logger.info("=" * 50)
        self.logger.info("시스템 하드웨어 정보")
        self.logger.info(cpu_info)
        self.logger.info(memory_info)
        self.logger.info(gpu_info)
        self.logger.info("=" * 50)
    
    def log_model_info(self, model_name, params_count, model_config=None):
        """모델 정보 로깅"""
        self.stats['model_params'][model_name] = params_count
        
        self.logger.info("=" * 50)
        self.logger.info(f"모델 정보: {model_name}")
        self.logger.info(f"파라미터 수: {params_count:,}")
        if model_config:
            self.logger.info(f"모델 구성: {json.dumps(model_config, indent=2)}")
        self.logger.info("=" * 50)
    
    def log_training_step(self, epoch, batch_idx, loss, batch_size, total_batches):
        """학습 스텝 로깅"""
        if batch_idx % max(1, total_batches // 10) == 0:
            self.stats['train_batch_time'].append(time.time())
            self.stats['batch_sizes'].append(batch_size)
            
            # GPU 사용량
            if torch.cuda.is_available():
                gpu_usage = []
                for i in range(torch.cuda.device_count()):
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    gpu_usage.append((reserved, allocated))
                self.stats['gpu_usage'].append(gpu_usage)
            
            # CPU 및 메모리 사용량
            self.stats['cpu_usage'].append(psutil.cpu_percent())
            self.stats['memory_usage'].append(psutil.virtual_memory().percent)
            
            self.logger.info(f"Epoch {epoch+1} - Batch {batch_idx}/{total_batches} - Loss: {loss:.6f}")
    
    def log_epoch_stats(self, epoch, train_loss, val_loss=None, metrics=None):
        """에포크 단위 통계 로깅"""
        self.stats['train_time'].append(time.time())
        
        log_msg = f"Epoch {epoch+1} 완료 - Train Loss: {train_loss:.6f}"
        if val_loss is not None:
            log_msg += f", Val Loss: {val_loss:.6f}"
        
        # 추가 메트릭이 있으면 로깅
        if metrics:
            for name, value in metrics.items():
                log_msg += f", {name}: {value:.6f}"
        
        self.logger.info(log_msg)
    
    def log_evaluation(self, metrics):
        """평가 결과 로깅"""
        self.stats['eval_time'].append(time.time())
        
        self.logger.info("=" * 50)
        self.logger.info("평가 결과:")
        for metric_name, value in metrics.items():
            self.logger.info(f"{metric_name}: {value:.6f}")
        self.logger.info("=" * 50)
    
    def log_prediction_time(self, region_count, future_steps):
        """예측 시간 로깅"""
        self.stats['predict_time'].append(time.time())
        
        elapsed = time.time() - self.start_time
        self.logger.info(f"예측 완료: {region_count}개 지역, {future_steps}일치, {elapsed:.2f}초 소요")
    
    def save_performance_summary(self):
        """성능 요약 저장"""
        # 훈련 시간 계산
        train_times = []
        if len(self.stats['train_time']) > 1:
            train_times = [self.stats['train_time'][i+1] - self.stats['train_time'][i] 
                         for i in range(len(self.stats['train_time'])-1)]
        
        total_time = time.time() - self.start_time
        
        # 요약 정보
        summary = {
            '총 실행 시간(초)': total_time,
            '평균 에포크 훈련 시간(초)': np.mean(train_times) if train_times else 0,
            '평균 CPU 사용률(%)': np.mean(self.stats['cpu_usage']) if self.stats['cpu_usage'] else 0,
            '평균 메모리 사용률(%)': np.mean(self.stats['memory_usage']) if self.stats['memory_usage'] else 0,
            '모델 파라미터 수': self.stats['model_params'],
            '평균 배치 크기': np.mean(self.stats['batch_sizes']) if self.stats['batch_sizes'] else 0
        }
        
        # GPU 통계 추가
        if self.stats['gpu_usage']:
            gpu_mean = np.mean([usage[0][1] for usage in self.stats['gpu_usage']])  # 첫 번째 GPU의 할당된 메모리
            summary['평균 GPU 메모리 사용량(GB)'] = gpu_mean
        
        # 요약 로깅
        self.logger.info("=" * 50)
        self.logger.info("성능 요약:")
        for key, value in summary.items():
            if isinstance(value, dict):
                self.logger.info(f"{key}:")
                for sub_key, sub_value in value.items():
                    self.logger.info(f"  {sub_key}: {sub_value:,}")
            else:
                self.logger.info(f"{key}: {value:.4f}")
        self.logger.info("=" * 50)
        
        # 요약 파일 저장
        summary_file = os.path.join(self.log_dir, f"performance_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 시각화 및 그래프 저장
        self._plot_performance_graphs()
    
    def _plot_performance_graphs(self):
        """성능 그래프 시각화 및 저장"""
        if not self.stats['train_time']:
            return
            
        # 시간 기반 x축 생성
        start_time = self.stats['train_time'][0]
        times = [(t - start_time) / 60 for t in self.stats['train_time']]  # 분 단위로 변환
        
        # 그래프 생성
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # CPU 및 메모리 사용량
        if self.stats['cpu_usage']:
            batch_times = [(t - start_time) / 60 for t in self.stats['train_batch_time']]
            axs[0, 0].plot(batch_times, self.stats['cpu_usage'], marker='o', linestyle='-', label='CPU')
            axs[0, 0].plot(batch_times, self.stats['memory_usage'], marker='s', linestyle='-', label='Memory')
            axs[0, 0].set_title('CPU & 메모리 사용률')
            axs[0, 0].set_xlabel('시간 (분)')
            axs[0, 0].set_ylabel('사용률 (%)')
            axs[0, 0].grid(True)
            axs[0, 0].legend()
        
        # GPU 메모리 사용량
        if self.stats['gpu_usage']:
            batch_times = [(t - start_time) / 60 for t in self.stats['train_batch_time']]
            gpu_allocated = [usage[0][1] for usage in self.stats['gpu_usage']]  # 첫 번째 GPU의 할당된 메모리
            axs[0, 1].plot(batch_times, gpu_allocated, marker='o', linestyle='-', color='green')
            axs[0, 1].set_title('GPU 메모리 사용량')
            axs[0, 1].set_xlabel('시간 (분)')
            axs[0, 1].set_ylabel('사용량 (GB)')
            axs[0, 1].grid(True)
        
        # 배치 크기
        if self.stats['batch_sizes']:
            batch_times = [(t - start_time) / 60 for t in self.stats['train_batch_time']]
            axs[1, 0].plot(batch_times, self.stats['batch_sizes'], marker='o', linestyle='-', color='purple')
            axs[1, 0].set_title('배치 크기')
            axs[1, 0].set_xlabel('시간 (분)')
            axs[1, 0].set_ylabel('배치 크기')
            axs[1, 0].grid(True)
        
        # 에포크별 훈련 시간
        if len(self.stats['train_time']) > 1:
            train_times = [self.stats['train_time'][i+1] - self.stats['train_time'][i] 
                         for i in range(len(self.stats['train_time'])-1)]
            epochs = range(1, len(train_times) + 1)
            axs[1, 1].plot(epochs, train_times, marker='o', linestyle='-', color='orange')
            axs[1, 1].set_title('에포크별 훈련 시간')
            axs[1, 1].set_xlabel('에포크')
            axs[1, 1].set_ylabel('시간 (초)')
            axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"performance_graphs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
        plt.close()

class CustomProgressBar:
    """진행 상황을 표시하는 커스텀 프로그레스 바"""
    
    def __init__(self, total, desc='진행 중', width=50):
        self.total = total
        self.desc = desc
        self.width = 50
        self.current = 0
        self.start_time = time.time()
        self.update(0)
    
    def update(self, progress):
        """진행도를 업데이트하고 표시합니다."""
        self.current += progress
        percentage = min(100, max(0, int(self.current / self.total * 100)))
        filled_length = int(self.width * percentage / 100)
        bar = '█' * filled_length + '-' * (self.width - filled_length)
        
        # 경과 시간
        elapsed = time.time() - self.start_time
        
        # 예상 남은 시간
        if percentage > 0:
            eta = elapsed / percentage * (100 - percentage)
            eta_str = self._format_time(eta)
        else:
            eta_str = "계산 중..."
        
        sys.stdout.write(f'\r{self.desc}: |{bar}| {percentage}% 완료 - 경과: {self._format_time(elapsed)} - 남은 시간: {eta_str}')
        sys.stdout.flush()
        
        # 100%에 도달하면 줄바꿈
        if percentage == 100:
            print()
    
    def _format_time(self, seconds):
        """초를 사람이 읽기 쉬운 형식으로 변환"""
        if seconds < 60:
            return f"{seconds:.1f}초"
        elif seconds < 3600:
            minutes, seconds = divmod(seconds, 60)
            return f"{int(minutes)}분 {int(seconds)}초"
        else:
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours)}시간 {int(minutes)}분"

class PrettyLogger:
    """보기 좋은 로그 포맷을 제공하는 로거 클래스"""
    
    def __init__(self, name, log_dir='logs', console_level=logging.INFO, file_level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 기존 핸들러 제거
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 디렉토리 생성
        os.makedirs(log_dir, exist_ok=True)
        
        # 콘솔 핸들러
        console = logging.StreamHandler()
        console.setLevel(console_level)
        
        # 파일 핸들러
        log_file = f"{log_dir}/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        
        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console)
        self.logger.addHandler(file_handler)
    
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)
    
    def section(self, title):
        """섹션 구분선 출력"""
        self.logger.info("\n" + "="*50)
        self.logger.info(f" {title} ".center(50, "="))
        self.logger.info("="*50)
    
    def subsection(self, title):
        """소섹션 구분선 출력"""
        self.logger.info("\n" + "-"*40)
        self.logger.info(f" {title} ")
        self.logger.info("-"*40)
    
    def result_table(self, df, title=None):
        """데이터프레임을 테이블 형태로 출력"""
        if title:
            self.subsection(title)
        
        # 데이터프레임을 문자열로 변환하여 로깅
        table_str = df.to_string()
        for line in table_str.split('\n'):
            self.logger.info(line)

def setup_logging(name="KAIM-ML", level=logging.INFO):
    """
    애플리케이션에 대한 로깅을 설정합니다.
    
    Args:
        name: 로거 이름
        level: 로깅 레벨
    
    Returns:
        구성된 로거 인스턴스
    """
    return PrettyLogger(name, console_level=level).logger

def tqdm_with_logging(iterable, logger, desc="진행 중", level=logging.INFO):
    """
    tqdm 진행 표시줄과 함께 로깅을 지원하는 래퍼
    
    Args:
        iterable: 반복 가능한 객체
        logger: 로깅에 사용할 로거
        desc: 진행 표시줄 설명
        level: 로깅 레벨
    
    Returns:
        tqdm 객체
    """
    # 시작 로깅
    logger.log(level, f"{desc} 시작...")
    
    # tqdm으로 감싸기
    with tqdm(iterable, desc=desc) as pbar:
        # 원본 업데이트 메서드 저장
        original_update = pbar.update
        
        # 업데이트 메서드 재정의
        def update_with_logging(n=1):
            original_update(n)
            if pbar.n == pbar.total:
                logger.log(level, f"{desc} 완료 ({pbar.format_dict['elapsed']:.2f}초)")
        
        pbar.update = update_with_logging
        yield from pbar

# 전역 성능 로거 인스턴스
performance_logger = None

def get_performance_logger():
    """전역 성능 로거 인스턴스 반환"""
    global performance_logger
    if performance_logger is None:
        performance_logger = PerformanceLogger()
    return performance_logger
