import os
import time
import psutil
import threading
import numpy as np
import matplotlib.pyplot as plt

try:
    import pynvml
    NVIDIA_SMI_AVAILABLE = True
except ImportError:
    NVIDIA_SMI_AVAILABLE = False

class ResourceMonitor:
    """시스템 리소스 모니터링 클래스"""
    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.cpu_percentages = []
        self.memory_percentages = []
        self.gpu_percentages = []
        self.timestamps = []
        self.start_time = None
        
        # GPU 모니터링 초기화
        if NVIDIA_SMI_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 첫 번째 GPU
            except:
                self.device_count = 0
        else:
            self.device_count = 0
    
    def get_gpu_utilization(self):
        """GPU 사용률 반환"""
        if NVIDIA_SMI_AVAILABLE and self.device_count > 0:
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                return utilization.gpu  # GPU 사용률 (0-100%)
            except:
                return 0
        return 0
    
    def monitor_thread(self):
        """리소스 모니터링 스레드"""
        self.start_time = time.time()
        
        while self.running:
            # 현재 시간
            current_time = time.time() - self.start_time
            self.timestamps.append(current_time)
            
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_percentages.append(cpu_percent)
            
            # 메모리 사용률
            memory_percent = psutil.virtual_memory().percent
            self.memory_percentages.append(memory_percent)
            
            # GPU 사용률
            gpu_percent = self.get_gpu_utilization()
            self.gpu_percentages.append(gpu_percent)
            
            # 간격 대기
            time.sleep(self.interval)
    
    def start(self):
        """모니터링 시작"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.monitor_thread)
            self.thread.daemon = True
            self.thread.start()
            print("리소스 모니터링 시작")
    
    def stop(self):
        """모니터링 중지"""
        if self.running:
            self.running = False
            self.thread.join()
            print("리소스 모니터링 중지")
    
    def plot(self, save_path=None):
        """모니터링 결과 시각화"""
        if not self.timestamps:
            print("모니터링 데이터가 없습니다.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # CPU 및 메모리 사용률
        ax.plot(self.timestamps, self.cpu_percentages, label='CPU 사용률 (%)')
        ax.plot(self.timestamps, self.memory_percentages, label='메모리 사용률 (%)')
        
        # GPU 사용률 (있을 경우)
        if NVIDIA_SMI_AVAILABLE and self.device_count > 0:
            ax.plot(self.timestamps, self.gpu_percentages, label='GPU 사용률 (%)')
        
        ax.set_xlabel('시간 (초)')
        ax.set_ylabel('사용률 (%)')
        ax.set_title('시스템 리소스 사용량')
        ax.grid(True)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path)
            print(f"결과가 {save_path}에 저장되었습니다.")
        else:
            plt.show()
        
        # 통계 출력
        print(f"평균 CPU 사용률: {np.mean(self.cpu_percentages):.1f}%")
        print(f"최대 CPU 사용률: {np.max(self.cpu_percentages):.1f}%")
        if NVIDIA_SMI_AVAILABLE and self.device_count > 0:
            print(f"평균 GPU 사용률: {np.mean(self.gpu_percentages):.1f}%")
            print(f"최대 GPU 사용률: {np.max(self.gpu_percentages):.1f}%")

# 사용 예시
if __name__ == "__main__":
    monitor = ResourceMonitor(interval=0.5)
    monitor.start()
    
    try:
        # 여기서 리소스 집약적인 작업을 수행
        print("모니터링 중... 중단하려면 Ctrl+C를 누르세요.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
        monitor.plot(save_path="resource_usage.png")
