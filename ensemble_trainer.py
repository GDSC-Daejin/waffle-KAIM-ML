import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model import create_model, EnsembleModel
from train import train_model, train_ensemble_models
from data_preprocessor import normalize_data, create_dataset, prepare_data, split_train_test, analyze_feature_importance
from utils import cache_result, parallel_process, get_optimal_device_config, memory_efficient_predict, parallel_train_for_regions
from tqdm import tqdm
import os
import pickle
import logging
from datetime import datetime, timedelta

class OilPriceEnsembleTrainer:
    """
    유가 예측을 위한 앙상블 모델 트레이너
    """
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
        self.device, self.use_mixed_precision, self.num_workers = get_optimal_device_config()
        
        # 로깅 설정
        self.setup_logger()
        
        # 초기 데이터 준비
        self.prepare_data()
        
        # 모델 구성
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
                "model_type": "transformer",
                "input_dim": self.X.shape[2],
                "hidden_dim": 64,
                "output_dim": len(self.target_indices),
                "nhead": 4,
                "num_layers": 2,
                "dropout": 0.3
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
        
        # 콘솔 핸들러
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 파일 핸들러
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(f"{log_dir}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh.setLevel(logging.INFO)
        
        # 포맷 설정
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
            features = self.features_df.drop(columns=['date'])
        else:
            self.dates = None
            features = self.features_df.copy()
        
        # 'area' 컬럼 처리
        if 'area' in features.columns:
            self.regions = features['area'].unique().tolist()
            self.logger.info(f"발견된 지역: {self.regions}")
            self.is_region_data = True
            
            # area 컬럼을 제거 (중요: 이 부분이 누락됨)
            features = features.drop(columns=['area'])
        else:
            self.regions = ['all']
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
        
        # 기본 LSTM 모델 학습
        input_dim = self.X_train.shape[2]
        hidden_dim = 64
        layer_dim = 2
        output_dim = len(self.target_indices)
        
        from model import LSTMModel
        model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
        model, _ = train_model(model, self.X_train, self.Y_train, epochs=100, batch_size=32, verbose=1)
        
        # SHAP 값을 사용한 특성 중요도 분석
        importance_df = analyze_feature_importance(
            model, 
            self.X_test.numpy(), 
            self.feature_names, 
            self.target_names
        )
        
        self.logger.info(f"각 타겟에 대한 상위 5개 특성:")
        for target in self.target_names:
            top_features = importance_df[target].sort_values(ascending=False).head(5)
            self.logger.info(f"{target}: {top_features.index.tolist()}")
        
        return importance_df

    @cache_result(expire_hours=48)
    def train_base_models(self, epochs=100, batch_size=32, patience=10):
        """기본 모델 학습"""
        self.logger.info("기본 모델 학습 시작...")
        
        # 앙상블 모델 학습
        models, weights = train_ensemble_models(
            self.model_configs,
            self.X_train,
            self.Y_train,
            ensemble_size=self.ensemble_size,
            use_gpu=self.use_gpu,
            n_jobs=self.num_workers,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )
        
        # 앙상블 모델 성능 평가
        ensemble = EnsembleModel(models, weights)
        
        X_test_tensor = torch.FloatTensor(self.X_test.numpy())
        pred = ensemble.predict(X_test_tensor)
        
        errors = {}
        for i, target in enumerate(self.target_names):
            mae = mean_absolute_error(self.Y_test.numpy()[:, i], pred[:, i])
            mse = mean_squared_error(self.Y_test.numpy()[:, i], pred[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(self.Y_test.numpy()[:, i], pred[:, i])
            
            errors[target] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            }
            
            self.logger.info(f"{target} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        return ensemble, errors

    def train_region_models(self, epochs=100, batch_size=None, patience=10):
        """지역별 모델 학습"""
        # GPU 최적화를 위한 배치 크기 자동 계산
        if batch_size is None:
            if self.use_gpu and torch.cuda.is_available():
                # GPU 메모리 크기에 따라 최적의 배치 크기 선택
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB 단위
                if gpu_mem > 10:  # 고용량 GPU (예: RTX 3080 이상)
                    batch_size = 256
                elif gpu_mem > 7:  # 중간 용량 GPU (예: GTX 1080)
                    batch_size = 128
                else:  # 저용량 GPU
                    batch_size = 64
            else:
                batch_size = 32  # CPU 기본값
        
        self.logger.info(f"학습에 사용할 배치 크기: {batch_size}")
        
        if not self.is_region_data:
            self.logger.info("지역 정보가 없어 전체 데이터로 단일 모델 학습 중...")
            ensemble, _ = self.train_base_models(epochs, batch_size, patience)
            self.region_models['all'] = ensemble
            self.target_scalers['all'] = self.scaler
            return
        
        self.logger.info(f"{len(self.regions)}개 지역에 대한 모델 학습 시작...")
        
        def train_for_region(region):
            # 지역별 데이터 필터링
            region_data = self.features_df[self.features_df['area'] == region].copy()
            
            # 충분한 데이터가 있는지 확인
            if len(region_data) < 30:
                self.logger.warning(f"지역 {region}의 데이터가 부족합니다. 건너뜁니다.")
                return None, None
            
            # 'date' 컬럼 제외
            if 'date' in region_data.columns:
                region_data = region_data.drop(columns=['date'])
            
            # 'area' 컬럼 제외
            if 'area' in region_data.columns:
                region_data = region_data.drop(columns=['area'])
            
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
            X_train, Y_train, X_test, Y_test = split_train_test(X, Y, self.test_size)
            
            # 모델 구성 조정
            model_configs = []
            for config in self.model_configs:
                new_config = config.copy()
                new_config['input_dim'] = X.shape[2]
                new_config['output_dim'] = len(target_indices)
                model_configs.append(new_config)
            
            # 앙상블 모델 학습
            models, weights = train_ensemble_models(
                model_configs,
                X_train,
                Y_train,
                ensemble_size=self.ensemble_size,
                use_gpu=self.use_gpu,
                n_jobs=1,  # 병렬 처리는 상위 수준에서 처리
                epochs=epochs,
                batch_size=batch_size,
                patience=patience
            )
            
            return EnsembleModel(models, weights), scaler
        
        # 병렬 처리
        results = []
        for region in tqdm(self.regions, desc="지역별 모델 학습"):
            ensemble, scaler = train_for_region(region)
            if ensemble is not None:
                self.region_models[region] = ensemble
                self.target_scalers[region] = scaler
                results.append(region)
        
        self.logger.info(f"{len(results)}개 지역에 대한 모델 학습 완료")

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
        
        for region, model in self.region_models.items():
            self.logger.info(f"지역 {region}에 대한 예측 수행 중...")
            
            # 마지막 시퀀스 데이터 가져오기
            if self.is_region_data:
                region_data = self.features_df[self.features_df['area'] == region].copy()
                if 'date' in region_data.columns:
                    region_data = region_data.drop(columns=['date'])
                if 'area' in region_data.columns:
                    region_data = region_data.drop(columns(['area']))
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
            
            for _ in range(future_steps):
                # 현재 시퀀스에서 예측
                input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
                pred = model.predict(input_tensor)[0]
                region_predictions.append(pred)
                
                # 시퀀스 업데이트: 마지막 항목 제거하고 예측값 추가
                new_row = np.zeros((1, current_sequence.shape[1]))
                if self.is_region_data:
                    target_indices = [region_data.columns.tolist().index(col) for col in self.target_names if col in region_data.columns]
                    for i, target_idx in enumerate(target_indices):
                        new_row[0, target_idx] = pred[i]
                else:
                    for i, target_idx in enumerate(self.target_indices):
                        new_row[0, target_idx] = pred[i]
                
                current_sequence = np.vstack([current_sequence[1:], new_row])
            
            # 예측 결과를 원래 스케일로 역변환
            if self.is_region_data:
                # 수정: 역변환 과정에서 인덱스 오류 가능성 확인
                try:
                    target_indices = [region_data.columns.tolist().index(col) for col in self.target_names if col in region_data.columns]
                    original_scale_preds = np.zeros((future_steps, len(target_indices)))
                    for i in range(future_steps):
                        temp = np.zeros((1, normalized_data.shape[1]))
                        for j, target_idx in enumerate(target_indices):
                            if j < len(region_predictions[i]):
                                temp[0, target_idx] = region_predictions[i][j]
                        temp = scaler.inverse_transform(temp)
                        for j, target_idx in enumerate(target_indices):
                            original_scale_preds[i, j] = temp[0, target_idx]
                except Exception as e:
                    self.logger.error(f"지역 {region} 예측 역변환 중 오류: {str(e)}")
                    continue
            else:
                original_scale_preds = np.zeros((future_steps, len(self.target_indices)))
                for i in range(future_steps):
                    temp = np.zeros((1, self.normalized_data.shape[1]))
                    for j, target_idx in enumerate(self.target_indices):
                        temp[0, target_idx] = region_predictions[i][j]
                    temp = scaler.inverse_transform(temp)
                    for j, target_idx in enumerate(self.target_indices):
                        original_scale_preds[i, j] = temp[0, target_idx]
            
            # 데이터프레임 생성
            pred_df = pd.DataFrame(
                original_scale_preds,
                columns=self.target_names,
                index=future_dates
            )
            
            predictions[region] = pred_df
        
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
            if os.path.isdir(os.path.join(models_dir, region_dir)):
                region = region_dir.replace('_', ' ')
                
                model_path = os.path.join(models_dir, region_dir, 'ensemble_model.pkl')
                scaler_path = os.path.join(models_dir, region_dir, 'scaler.pkl')
                
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
            self.logger.info(f"{len(loaded_regions)}개 지역의 모델을 로드했습니다: {loaded_regions}")
            return True
        else:
            self.logger.warning("로드할 모델이 없습니다.")
            return False

def run_ensemble_prediction_pipeline(features_df, target_cols, look_back=3, future_steps=7, 
                                     ensemble_size=3, use_gpu=True, batch_size=None):
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
    # 앙상블 트레이너 초기화
    trainer = OilPriceEnsembleTrainer(
        features_df=features_df,
        target_cols=target_cols,
        look_back=look_back,
        ensemble_size=ensemble_size,
        use_gpu=use_gpu
    )
    
    # 저장된 모델이 있는지 확인하고 로드
    if not trainer.load_models():
        # 변수 중요도 분석
        importance_df = trainer.analyze_variable_importance()
        
        # 지역별 모델 학습 - batch_size 매개변수 전달
        trainer.train_region_models(epochs=100, batch_size=batch_size, patience=10)
        
        # 학습된 모델 저장
        trainer.save_models()
    
    # 미래 예측 수행
    predictions = trainer.predict_future(future_steps=future_steps)
    
    return predictions