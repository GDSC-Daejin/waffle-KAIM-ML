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
        log_file = os.path.join(log_dir, f"performance_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
        cpu_count = os.cpu_count()
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

# 전역 성능 로거 인스턴스
performance_logger = None

def get_performance_logger():
    """전역 성능 로거 인스턴스 반환"""
    global performance_logger
    if performance_logger is None:
        performance_logger = PerformanceLogger()
    return performance_logger
