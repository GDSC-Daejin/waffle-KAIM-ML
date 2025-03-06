import torch
import numpy as np
import pandas as pd
import json
import time
import logging
import os
import psutil
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model import create_model, EnsembleModel
from train import train_model
from data_preprocessor import normalize_data, create_dataset, prepare_data, split_train_test, analyze_feature_importance
from utils import cache_result, optimize_memory_usage
from parallel_utils import optimize_gpu_tensor_ops
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import sys
from dotenv import load_dotenv  # dotenv 추가

# .env 파일 로드
load_dotenv()

# 환경 변수 값 읽기
EPOCHS = int(os.getenv("EPOCHS", "100"))
IMPORTANCE_EPOCHS = int(os.getenv("IMPORTANCE_EPOCHS", "50"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
PATIENCE = int(os.getenv("PATIENCE", "10"))
ENSEMBLE_SIZE = int(os.getenv("ENSEMBLE_SIZE", "3"))

class OilPriceEnsembleTrainer:
    """유가 예측을 위한 앙상블 모델 트레이너"""
    def __init__(self, features_df, target_cols, look_back=3, test_size=0.2,
                ensemble_size=5, use_gpu=True, cache_dir='model_cache'):
        """
        초기화
        
        Args:
            features_df: 특성 데이터프레임
            target_cols: 타겟 컬럼 리스트
            look_back: 시계열 윈도우 크기
            test_size: 테스트 데이터 비율
            ensemble_size: 앙상블에 포함할 모델 수
            use_gpu: GPU 사용 여부
            cache_dir: 모델 캐시 디렉토리
        """
        self.features_df = features_df
        self.target_cols = target_cols
        self.look_back = look_back
        self.test_size = test_size
        self.ensemble_size = ensemble_size
        self.use_gpu = use_gpu
        
        # 모델 캐시 디렉토리 생성
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 최적 디바이스 구성
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.use_mixed_precision = torch.cuda.is_available() and use_gpu
        
        # 시스템 리소스에 맞게 최적화 (E5-2683v4 CPU 고려)
        cpu_count = os.cpu_count() or 16
        self.num_workers = min(cpu_count - 2, 32)  # CPU 코어 여유분 남김
        
        # 메모리 최적화 (128GB RAM 고려)
        optimize_memory_usage()
        
        # 로깅 설정
        self.setup_logger()
        self.logger.info(f"시스템 리소스: CPU {cpu_count}코어, GPU {torch.cuda.device_count()}개, 사용 워커: {self.num_workers}개")
        
        # 초기 데이터 준비
        self.prepare_data()
        
        # 모델 구성 - 다양한 앙상블 구성
        self.model_configs = [
            {
                "model_type": "lstm",
                "input_dim": self.X.shape[2],
                "hidden_dim": 64,
                "layer_dim": 2,
                "output_dim": len(self.target_indices),
                "dropout": 0.3
            },
            {
                "model_type": "gru",
                "input_dim": self.X.shape[2],
                "hidden_dim": 64,
                "layer_dim": 2,
                "output_dim": len(self.target_indices),
                "dropout": 0.3
            },
            {
                "model_type": "cnn",
                "input_dim": self.X.shape[2],
                "hidden_dim": 64,
                "output_dim": len(self.target_indices),
                "sequence_length": self.look_back,
                "dropout": 0.3
            },
            {
                "model_type": "lstm",  # 다양한 하이퍼파라미터로 추가 LSTM 모델
                "input_dim": self.X.shape[2],
                "hidden_dim": 128,
                "layer_dim": 1,
                "output_dim": len(self.target_indices),
                "dropout": 0.2
            },
            {
                "model_type": "gru",  # 다른 구성의 GRU
                "input_dim": self.X.shape[2],
                "hidden_dim": 96,
                "layer_dim": 3,
                "output_dim": len(self.target_indices),
                "dropout": 0.4
            }
        ]
        
        # 각 지역에 대한 앙상블 모델 저장
        self.region_models = {}
        # 타겟 스케일러 저장
        self.target_scalers = {}

    def setup_logger(self):
        """로깅 설정"""
        self.logger = logging.getLogger("OilPriceEnsemble")
        self.logger.setLevel(logging.INFO)
        
        # 핸들러가 이미 있는지 확인
        if not self.logger.handlers:
            # 콘솔 핸들러
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # 파일 핸들러
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(f"{log_dir}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            fh.setLevel(logging.INFO)
            
            # 포맷 설정 수정 - %(level)s를 %(levelname)s로 변경
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

    def prepare_data(self):
        """데이터 전처리 및 준비"""
        self.logger.info("데이터 전처리 시작...")
        
        # 'date' 컬럼 제외 (날짜는 별도 처리)
        if 'date' in self.features_df.columns:
            self.dates = self.features_df['date'].copy()
            features = self.features_df.drop(['date'], axis=1)
        else:
            self.dates = None
            features = self.features_df.copy()
        
        # 'area' 컬럼 처리
        if 'area' in features.columns:
            self.regions = features['area'].unique().tolist()
            self.logger.info(f"발견된 지역: {self.regions}")
            self.is_region_data = True
            
            # area 컬럼을 제거ㄴㄴ
            features = features.drop(['area'], axis=1)
        else:
            self.regions = ['National']
            self.is_region_data = False
        
        # 타겟 컬럼 식별
        valid_targets = [col for col in self.target_cols if col in features.columns]
        if not valid_targets:
            raise ValueError("유효한 타겟 컬럼이 없습니다.")
        
        self.target_indices = [features.columns.get_loc(col) for col in valid_targets]
        self.feature_names = features.columns.tolist()
        self.target_names = valid_targets
        
        self.logger.info(f"타겟 변수: {self.target_names}")
        self.logger.info(f"특성 수: {len(self.feature_names)}")
        
        # 전체 데이터에 대한 정규화 및 시계열 데이터셋 생성
        self.data_array = features.values.astype(float)
        self.normalized_data, self.scaler = normalize_data(self.data_array)
        
        # 시계열 데이터셋 생성
        X, Y_full = create_dataset(self.normalized_data, self.look_back)
        self.X, self.Y_full = prepare_data(X, Y_full)
        
        # 타겟 변수만 선택
        self.Y = self.Y_full[:, self.target_indices]
        
        # 학습/테스트 분할
        self.X_train, self.Y_train, self.X_test, self.Y_test = split_train_test(self.X, self.Y, self.test_size)
        
        self.logger.info(f"데이터 분할 완료 - 학습 샘플: {len(self.X_train)}, 테스트 샘플: {len(self.X_test)}")
        self.logger.info("데이터 전처리 완료")

    @cache_result(expire_hours=48)
    def analyze_variable_importance(self):
        """변수 중요도 분석"""
        self.logger.info("변수 중요도 분석 시작...")
        
        try:
            # 기본 LSTM 모델 학습
            input_dim = self.X_train.shape[2]
            hidden_dim = 64
            layer_dim = 2
            output_dim = len(self.target_indices)
            
            from model import LSTMModel
            model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
            
            # 모델을 적절한 디바이스로 이동
            model = model.to(self.device)
            
            # 학습 데이터를 디바이스로 이동
            X_train_device = self.X_train.to(self.device)
            Y_train_device = self.Y_train.to(self.device)
            
            # .env에서 설정한 에포크 수 사용
            self.logger.info(f"변수 중요도 분석을 위해 {IMPORTANCE_EPOCHS}번의 에포크로 모델 학습")
            model, _ = train_model(model, X_train_device, Y_train_device, 
                              epochs=IMPORTANCE_EPOCHS, batch_size=BATCH_SIZE, verbose=10)
            
            # 중요: 분석 전에 CPU로 모델 이동
            model = model.cpu()
            
            # 중요: X_test도 CPU로 이동 후 numpy 변환
            X_test_cpu = self.X_test.cpu().numpy() if torch.is_tensor(self.X_test) else self.X_test
            
            # 특성 중요도 분석
            importance_df = analyze_feature_importance(
                model, 
                X_test_cpu,
                self.feature_names, 
                self.target_names
            )
            
            # 상위 5개 특성 출력
            self.logger.info(f"각 타겟에 대한 상위 5개 특성:")
            for target in self.target_names:
                top_features = importance_df[target].sort_values(ascending=False).head(5)
                self.logger.info(f"{target}: {top_features.index.tolist()}")
                
            return importance_df
            
        except Exception as e:
            self.logger.warning(f"특성 중요도 분석 중 오류 발생: {str(e)}. 대체 방법 사용")
            import traceback
            self.logger.warning(traceback.format_exc())
            
            # 대체 방법: 모든 특성에 동일한 중요도 부여
            return pd.DataFrame(
                np.ones((len(self.feature_names), len(self.target_names))), 
                index=self.feature_names, 
                columns=self.target_names
            )

    def train_region_models(self, epochs=None, batch_size=None, patience=None):
        """지역별 모델 학습"""
        # 기본값이 None인 경우 .env 설정 사용
        if epochs is None:
            epochs = EPOCHS
        if batch_size is None:
            if self.use_gpu and torch.cuda.is_available():
                batch_size = optimize_gpu_tensor_ops(mixed_precision=self.use_mixed_precision)
            else:
                # 128GB RAM 활용을 위한 CPU 배치 크기
                total_ram = psutil.virtual_memory().total / (1024**3)  # GB
                batch_size = min(2048, int(total_ram / 16))  # RAM 크기에 맞춤
        if patience is None:
            patience = PATIENCE
        
        self.logger.info(f"학습 설정: 에포크 {epochs}, 배치 크기 {batch_size}, 인내심 {patience}")
        
        # 단일 데이터셋 처리
        if not self.is_region_data:
            self.logger.info("지역 정보가 없어 전체 데이터로 단일 모델 학습 중...")
            ensemble = self._train_models_for_region(self.X_train, self.Y_train, epochs, batch_size, patience)
            self.region_models['National'] = ensemble
            self.target_scalers['National'] = self.scaler
            return
        
        # 지역별 모델 학습
        self.logger.info(f"{len(self.regions)}개 지역에 대한 모델 학습 시작...")
        
        for idx, region in enumerate(self.regions):
            # 진행 상태 표시
            progress = f"[{idx+1}/{len(self.regions)}]"
            self.logger.info(f"{progress} 지역 {region} 모델 학습 시작...")
            
            # 지역별 데이터 필터링
            region_data = self.features_df[self.features_df['area'] == region].copy()
            
            # 충분한 데이터가 있는지 확인
            if len(region_data) < 30:
                self.logger.warning(f"{progress} 지역 {region}의 데이터가 부족합니다. 건너뜁니다.")
                continue
            
            # 'date' 컬럼 제외
            if 'date' in region_data.columns:
                region_data = region_data.drop(['date'], axis=1)
            
            # 'area' 컬럼 제외
            if 'area' in region_data.columns:
                region_data = region_data.drop(['area'], axis=1)
            
            # 타겟 인덱스 조정
            feature_names = region_data.columns.tolist()
            target_indices = [feature_names.index(col) for col in self.target_names if col in feature_names]
            
            # 데이터 정규화 및 시계열 데이터셋 생성
            data_array = region_data.values.astype(float)
            normalized_data, scaler = normalize_data(data_array)
            
            X, Y_full = create_dataset(normalized_data, self.look_back)
            X, Y_full = prepare_data(X, Y_full)
            
            # 타겟 변수만 선택
            Y = Y_full[:, target_indices]
            
            # 학습/테스트 분할
            X_train, Y_train, _, _ = split_train_test(X, Y, self.test_size)
            
            # 모델 구성 조정
            model_configs = []
            for config in self.model_configs:
                new_config = config.copy()
                new_config['input_dim'] = X.shape[2]
                new_config['output_dim'] = len(target_indices)
                model_configs.append(new_config)
            
            # 앙상블 모델 학습
            self.logger.info(f"{progress} {region} 지역 앙상블 모델 학습 시작")
            ensemble = self._train_models_for_region(X_train, Y_train, epochs, batch_size, patience, model_configs)
            self.logger.info(f"{progress} {region} 지역 앙상블 모델 학습 완료")
            
            # 결과 저장
            self.region_models[region] = ensemble
            self.target_scalers[region] = scaler
            
            # 메모리 정리
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 모델을 저장하여 OOM 방지
            if idx % 5 == 0 and idx > 0:
                self.save_models()
                self.logger.info(f"{idx}번째 지역까지 모델 저장 완료")
            
        self.logger.info(f"{len(self.region_models)}개 지역에 대한 모델 학습 완료")

    def _train_models_for_region(self, X_train, Y_train, epochs=100, batch_size=32, patience=10, model_configs=None):
        """한 지역에 대한 앙상블 모델 학습"""
        if model_configs is None:
            model_configs = self.model_configs
            
        models = []
        weights = []
        val_losses = []
        
        # 검증 데이터 분할 (20%)
        train_size = int(len(X_train) * 0.8)
        X_train_data, X_val = X_train[:train_size], X_train[train_size:]
        Y_train_data, Y_val = Y_train[:train_size], Y_train[train_size:]
        
        # 안전한 배치 크기 설정 (GPU 메모리 과부하 방지)
        if self.use_gpu and torch.cuda.is_available():
            # GTX 1080 8GB에 더 안전한 배치 사이즈
            batch_size = min(batch_size, 12)
            self.logger.info(f"GPU 메모리 안전을 위해 배치 크기를 {batch_size}로 조정했습니다.")
        
        # 다양한 모델 훈련
        for i, config in enumerate(model_configs[:self.ensemble_size]):
            model_type = config.get('model_type', 'lstm')
            
            self.logger.info(f"모델 {i+1}/{len(model_configs[:self.ensemble_size])} ({model_type}) 훈련 중...")
            
            try:
                # 훈련 시작 전 GPU 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # 모델 생성 및 디바이스 이동
                model = create_model(**config)
                model = model.to(self.device)
                
                # 텐서 변환 및 디바이스 이동
                if isinstance(X_train_data, torch.Tensor):
                    X_train_tensor = X_train_data.to(self.device)
                else:
                    X_train_tensor = torch.FloatTensor(X_train_data).to(self.device)
                    
                if isinstance(Y_train_data, torch.Tensor):
                    Y_train_tensor = Y_train_data.to(self.device)
                else:
                    Y_train_tensor = torch.FloatTensor(Y_train_data).to(self.device)
                
                # 검증 데이터도 준비 (하지만 아직 디바이스로 이동하지 않음)
                if isinstance(X_val, torch.Tensor):
                    X_val_cpu = X_val.cpu()  # 일단 CPU에 보관
                else:
                    X_val_cpu = torch.FloatTensor(X_val)
                    
                if isinstance(Y_val, torch.Tensor):
                    Y_val_cpu = Y_val.cpu()  # 일단 CPU에 보관
                else:
                    X_val_cpu = torch.FloatTensor(Y_val)
                
                # 모델 학습 (수정된 train_model은 내부에서 디바이스 처리)
                model, history = train_model(
                    model, 
                    X_train_tensor, 
                    Y_train_tensor, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    verbose=5
                )
                
                # 검증 손실 계산 - 메모리 관리를 위해 작은 배치로 처리
                model.eval()
                val_loss = 0.0
                val_batch_size = 32  # 검증은 더 작은 배치로
                n_batches = 0
                
                with torch.no_grad():
                    for start_idx in range(0, len(X_val_cpu), val_batch_size):
                        end_idx = min(start_idx + val_batch_size, len(X_val_cpu))
                        
                        # 현재 배치만 GPU로 이동
                        X_batch = X_val_cpu[start_idx:end_idx].to(self.device)
                        Y_batch = Y_val_cpu[start_idx:end_idx].to(self.device)
                        
                        # 예측 및 손실 계산
                        outputs = model(X_batch)
                        batch_loss = torch.nn.MSELoss()(outputs, Y_batch).item()
                        
                        # 누적
                        val_loss += batch_loss
                        n_batches += 1
                        
                        # 배치 데이터 메모리 해제
                        del X_batch, Y_batch
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # 평균 손실 계산
                val_loss = val_loss / max(1, n_batches)
                val_losses.append(val_loss)
                
                # 검증 손실 기반 가중치 계산
                weight = 1.0 / (val_loss + 1e-10)  # 0으로 나누기 방지
                weights.append(weight)
                
                # CPU로 이동하여 메모리 절약
                model = model.cpu()
                models.append(model)
                
                self.logger.info(f"모델 {i+1} 학습 완료: 최종 검증 손실 {val_loss:.6f}")
                
            except Exception as e:
                self.logger.error(f"모델 {model_type} 학습 중 오류 발생: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                # 오류 발생 시 다음 모델로 계속 진행
                continue
            finally:
                # 메모리 정리 (finally 블록에서 항상 실행)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 학습된 모델이 없으면 오류 발생
        if not models:
            raise ValueError("모든 모델 학습에 실패했습니다.")
        
        # 가중치 계산 방식 개선
        if models:
            # 지수 가중치 적용
            weights = np.array(weights)
            weights = weights / weights.sum()  # 정규화
            self.logger.info(f"모델 가중치: {weights.round(3)}")
            
            # 앙상블 모델 생성
            ensemble = EnsembleModel(models, weights)
        
        return ensemble

    def predict_future(self, future_steps=7):
        """미래 예측 수행"""
        self.logger.info(f"향후 {future_steps}일에 대한 예측 시작...")
        
        # 마지막 날짜 확인
        if self.dates is not None:
            last_date = self.dates.iloc[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_steps)
        else:
            future_dates = [f"Day+{i+1}" for i in range(future_steps)]
        
        predictions = {}
        total_regions = len(self.region_models)
        
        for idx, (region, model) in enumerate(self.region_models.items()):
            # 진행 표시줄
            progress_bar = f"[{idx+1}/{total_regions}]"
            self.logger.info(f"{progress_bar} 지역 {region}에 대한 예측 수행 중...")
            
            try:
                # CPU에서 예측 수행
                model = model.cpu()
                
                # 마지막 시퀀스 데이터 가져오기
                if self.is_region_data:
                    region_data = self.features_df[self.features_df['area'] == region].copy()
                    if 'date' in region_data.columns:
                        region_data = region_data.drop(['date'], axis=1)
                    if 'area' in region_data.columns:
                        region_data = region_data.drop(['area'], axis=1)
                    data_array = region_data.values.astype(float)
                    normalized_data, _ = normalize_data(data_array)
                else:
                    normalized_data = self.normalized_data
                
                # 마지막 시퀀스
                last_sequence = normalized_data[-self.look_back:]
                
                # 예측 수행
                scaler = self.target_scalers[region]
                current_sequence = last_sequence.copy()
                region_predictions = []
                
                # 단계별 예측
                for step in range(future_steps):
                    # 현재 시퀀스로 예측
                    input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)  # 배치 차원 추가
                    
                    # 예측
                    with torch.no_grad():
                        pred = model(input_tensor).detach().cpu().numpy()[0]  # 배치 차원 제거
                        
                    region_predictions.append(pred)
                    
                    # 피드백 루프: 마지막 행 제거하고 예측값 추가
                    new_row = np.zeros((1, current_sequence.shape[1]))
                    
                    if self.is_region_data:
                        target_indices = [region_data.columns.tolist().index(col) for col in self.target_names if col in region_data.columns]
                        for i, target_idx in enumerate(target_indices):
                            if i < len(pred):
                                new_row[0, target_idx] = pred[i]
                    else:
                        for i, target_idx in enumerate(self.target_indices):
                            if i < len(pred):
                                new_row[0, target_idx] = pred[i]
                    
                    current_sequence = np.vstack([current_sequence[1:], new_row])
                
                # 예측 결과를 원래 스케일로 역변환
                original_scale_preds = []
                for step_pred in region_predictions:
                    # 전체 피처 차원으로 확장
                    full_pred = np.zeros((1, scaler.n_features_in_))
                    if self.is_region_data:
                        target_indices = [region_data.columns.tolist().index(col) for col in self.target_names if col in region_data.columns]
                        for i, target_idx in enumerate(target_indices):
                            if i < len(step_pred):
                                full_pred[0, target_idx] = step_pred[i]
                    else:
                        for i, target_idx in enumerate(self.target_indices):
                            if i < len(step_pred):
                                full_pred[0, target_idx] = step_pred[i]
                    
                    # 역변환
                    inverse_pred = scaler.inverse_transform(full_pred)[0]
                    
                    # 타겟 값만 추출
                    if self.is_region_data:
                        target_indices = [region_data.columns.tolist().index(col) for col in self.target_names if col in region_data.columns]
                        original_scale_preds.append([inverse_pred[idx] for idx in target_indices])
                    else:
                        original_scale_preds.append([inverse_pred[idx] for idx in self.target_indices])
                
                # 데이터프레임 생성
                pred_df = pd.DataFrame(
                    original_scale_preds,
                    columns=self.target_names,
                    index=future_dates
                )
                
                # 예측 결과 후처리 및 합리성 확인
                for col in pred_df.columns:
                    curr_vals = pred_df[col].values
                    for i in range(1, len(curr_vals)):
                        change_pct = abs((curr_vals[i] - curr_vals[i-1]) / curr_vals[i-1]) * 100
                        if change_pct > 10:  # 10% 이상 급변
                            # 이전 값과 다음 값의 평균으로 보정
                            if i < len(curr_vals) - 1:
                                curr_vals[i] = (curr_vals[i-1] + curr_vals[i+1]) / 2
                            else:
                                curr_vals[i] = curr_vals[i-1]  # 마지막 값은 이전 값으로
                    
                    pred_df[col] = curr_vals
                
                # 결과 저장
                predictions[region] = pred_df
                
                # 결과 요약 출력
                summary_stats = pred_df.describe().loc[['mean', 'min', 'max']]
                self.logger.info(f"{progress_bar} 지역 {region} 예측 완료")
                
                # NumPy float64를 일반 float로 변환하여 로그 출력 - 소수점 1자리로 제한
                mean_values = {k: round(float(v), 1) for k, v in dict(summary_stats.loc['mean']).items()}
                self.logger.info(f"평균 가격: {mean_values}")
                
            except Exception as e:
                self.logger.error(f"지역 {region} 예측 중 오류 발생: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        self.logger.info(f"총 {len(predictions)}개 지역에 대한 예측 완료")
        return predictions

    def save_models(self):
        """학습된 모델 저장"""
        models_dir = os.path.join(self.cache_dir, 'trained_models')
        os.makedirs(models_dir, exist_ok=True)
        
        # 메타데이터 저장
        metadata = {
            'date_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ensemble_size': self.ensemble_size,
            'look_back': self.look_back,
            'target_names': self.target_names,
            'regions': list(self.region_models.keys()),
            'feature_names': self.feature_names
        }
        
        meta_path = os.path.join(models_dir, 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 각 지역별 모델 저장
        for region, model in self.region_models.items():
            region_dir = os.path.join(models_dir, region.replace(' ', '_'))
            os.makedirs(region_dir, exist_ok=True)
            
            # 모델 저장
            model_path = os.path.join(region_dir, 'ensemble_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # 스케일러 저장
            scaler_path = os.path.join(region_dir, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.target_scalers[region], f)
        
        self.logger.info(f"모든 모델이 {models_dir}에 저장되었습니다.")

    def load_models(self):
        """저장된 모델 로드"""
        models_dir = os.path.join(self.cache_dir, 'trained_models')
        if not os.path.exists(models_dir):
            self.logger.warning("저장된 모델을 찾을 수 없습니다.")
            return False
        
        # 모델 로드
        loaded_regions = []
        for region_dir in os.listdir(models_dir):
            region_path = os.path.join(models_dir, region_dir)
            if os.path.isdir(region_path) and not region_dir.startswith('.'):
                region = region_dir.replace('_', ' ')
                
                model_path = os.path.join(region_path, 'ensemble_model.pkl')
                scaler_path = os.path.join(region_path, 'scaler.pkl')
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    try:
                        with open(model_path, 'rb') as f:
                            self.region_models[region] = pickle.load(f)
                        
                        with open(scaler_path, 'rb') as f:
                            self.target_scalers[region] = pickle.load(f)
                        
                        loaded_regions.append(region)
                    except Exception as e:
                        self.logger.error(f"지역 {region} 모델 로드 중 오류: {str(e)}")
        
        if loaded_regions:
            self.logger.info(f"{len(loaded_regions)}개 지역의 모델을 로드했습니다: {', '.join(loaded_regions[:5])}..." if len(loaded_regions) > 5 else loaded_regions)
            return True
        else:
            self.logger.warning("로드할 모델이 없습니다.")
            return False

    def plot_predictions(self, predictions, save_path=None):
        """예측 결과를 시각화"""
        # 경고 메시지 숨기기
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
        
        # 기본 matplotlib 설정만 사용
        import matplotlib as mpl
        mpl.rcParams['axes.unicode_minus'] = False
        
        for region, pred_df in predictions.items():
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for col in pred_df.columns:
                ax.plot(pred_df.index, pred_df[col], marker='o', linewidth=2, label=col)
            
            # 영어로만 제목과 라벨 설정
            ax.set_title(f"Future Oil Price Prediction - {region}", fontsize=15)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Price (KRW)", fontsize=12)
            ax.grid(True)
            ax.legend()
            
            plt.tight_layout()
            
            if save_path:
                region_save_path = os.path.join(save_path, f"{region.replace(' ', '_')}_prediction.png")
                plt.savefig(region_save_path)
                self.logger.info(f"{region} 예측 그래프 저장: {region_save_path}")
            else:
                plt.show()
            
            plt.close()


def run_ensemble_prediction_pipeline(features_df, target_cols, look_back=3, future_steps=7, 
                                    ensemble_size=None, use_gpu=True, batch_size=None):
    """
    앙상블 예측 파이프라인 실행
    
    Args:
        features_df: 특성 데이터프레임
        target_cols: 타겟 컬럼 리스트
        look_back: 시계열 윈도우 크기
        future_steps: 예측할 미래 일 수
        ensemble_size: 앙상블에 포함할 모델 수
        use_gpu: GPU 사용 여부
        batch_size: 학습 배치 크기 (None이면 자동 설정)
    
    Returns:
        지역별 예측 결과
    """
    # 로거 설정
    logger = logging.getLogger("EnsemblePipeline")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        # 포맷 설정 수정 - %(level)s를 %(levelname)s로 변경
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # 진행 상황 표시를 위한 상수 정의
    TOTAL_STEPS = 5  # 전체 단계 수
    current_step = 0
    
    def show_progress(step_description, step=1):
        nonlocal current_step
        current_step += step
        progress_bar = '=' * current_step + '>' + ' ' * (TOTAL_STEPS - current_step - 1)
        logger.info(f"\n[{progress_bar}] {current_step}/{TOTAL_STEPS} {step_description}")
    
    # 현재 시간 기록 (총 실행시간 계산용)
    start_time = time.time()
    
    # 앙상블 트레이너 초기화
    trainer = OilPriceEnsembleTrainer(
        features_df=features_df,
        target_cols=target_cols,
        look_back=look_back,
        ensemble_size=ensemble_size,
        use_gpu=use_gpu
    )
    
    show_progress("데이터 로드 및 전처리 완료")
    
    # 저장된 모델이 있는지 확인하고 로드
    if trainer.load_models():
        logger.info("저장된 모델을 성공적으로 로드했습니다.")
        show_progress("저장된 모델 로드", 3)  # 3단계 건너뜀
    else:
        logger.info("저장된 모델을 찾을 수 없거나 로드하는데 실패했습니다. 새로 학습합니다.")
        
        # 변수 중요도 분석
        try:
            importance_df = trainer.analyze_variable_importance()
            logger.info("특성 중요도 분석 완료")
            
            # 상위 5개 특성 출력
            for target in target_cols:
                if target in importance_df.columns:
                    top_features = importance_df[target].sort_values(ascending=False).head(5).index.tolist()
                    logger.info(f"{target}에 대한 상위 5개 특성: {', '.join(top_features)}")
        except Exception as e:
            logger.warning(f"특성 중요도 분석 중 오류 발생: {str(e)}. 모든 특성을 사용합니다.")
        
        show_progress("특성 중요도 분석 완료")
        
        # 지역별 모델 학습 시작 시간 기록
        model_start_time = time.time()
        logger.info("지역별 모델 학습 시작...")
        
        # 지역별 모델 학습
        trainer.train_region_models(epochs=EPOCHS, batch_size=batch_size, patience=PATIENCE)
        
        # 학습 완료 시간 및 소요 시간 출력
        model_end_time = time.time()
        model_elapsed_time = model_end_time - model_start_time
        hours, remainder = divmod(model_elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"모든 모델 학습 완료. 총 소요 시간: {int(hours)}시간 {int(minutes)}분 {seconds:.1f}초")
        
        show_progress("모델 학습 완료")
        
        # 학습된 모델 저장
        logger.info("학습된 모델 저장 중...")
        trainer.save_models()
        logger.info("모델 저장 완료")
        
        show_progress("모델 저장 완료")
    
    # 미래 예측 수행
    logger.info(f"향후 {future_steps}일에 대한 예측 시작...")
    predictions = trainer.predict_future(future_steps=future_steps)
    logger.info("예측 완료")
    
    # 예측 결과 시각화 (기본적으로 비활성화)
    try:
        # 결과 저장 디렉토리 생성
        images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_images')
        os.makedirs(images_dir, exist_ok=True)
        
        # 예측 결과 시각화 및 저장
        trainer.plot_predictions(predictions, save_path=images_dir)
        logger.info(f"예측 시각화 결과가 {images_dir}에 저장되었습니다.")
    except Exception as e:
        logger.warning(f"예측 시각화 중 오류 발생: {str(e)}")
    
    show_progress("예측 완료")
    
    # 총 실행 시간 출력
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"전체 예측 파이프라인 완료. 총 소요 시간: {int(hours)}시간 {int(minutes)}분 {seconds:.1f}초")
    
    # 결과 요약 출력
    print_prediction_summary(predictions, future_steps)
    
    return predictions

def print_prediction_summary(predictions, future_steps):
    """예측 결과 요약을 깔끔하게 출력합니다."""
    logger = logging.getLogger("EnsemblePipeline")
    
    logger.info("\n" + "="*50)
    logger.info(f"향후 {future_steps}일 예측 결과 요약")
    logger.info("="*50)
    
    # 지역별로 예측 요약
    for region, pred_df in predictions.items():
        logger.info(f"\n[지역: {region}]")
        
        # 유가 종류별 평균 가격 및 변동폭 출력
        for col in pred_df.columns:
            # NumPy float64를 일반 float로 변환
            avg_price = float(pred_df[col].mean())
            min_price = float(pred_df[col].min())
            max_price = float(pred_df[col].max())
            change = float(pred_df[col].iloc[-1] - pred_df[col].iloc[0])
            change_pct = float((change / pred_df[col].iloc[0]) * 100 if pred_df[col].iloc[0] != 0 else 0)
            logger.info(f"  {col}: 평균 {avg_price:.1f}원, 변동폭 {change:+.1f}원 ({change_pct:+.1f}%)")
        
        # 날짜별 상세 예측 결과 출력
        logger.info("\n  [일자별 예측 결과]")
        for date, row in pred_df.iterrows():
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            logger.info(f"  {date_str}:")
            
            for col in pred_df.columns:
                price = float(row[col])  # NumPy float64를 일반 float로 변환
                logger.info(f"    - {col}: {price:.1f}원")
    
    logger.info("\n" + "="*50)

if __name__ == "__main__":
    # 직접 실행 시 간단한 테스트 수행
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        from data_loader import load_data_from_mongo
        
        # 데이터 로드
        logger.info("데이터 로딩 중...")
        df = load_data_from_mongo()
        
        if df.empty:
            logger.error("데이터를 로드할 수 없습니다.")
            exit(1)
            
        logger.info(f"데이터 로드 완료: {df.shape[0]}개 레코드")
        
        # 타겟 컬럼 설정
        target_cols = ["gasoline", "premiumGasoline", "diesel", "kerosene"]
        
        # 예측 파이프라인 실행
        predictions = run_ensemble_prediction_pipeline(
            df, 
            target_cols, 
            look_back=3, 
            future_steps=7,
            ensemble_size=3,
            use_gpu=torch.cuda.is_available()
        )
        
        logger.info("예측 완료!")
        
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}", exc_info=True)