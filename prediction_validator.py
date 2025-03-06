
"""
현실적인 유가 예측을 위한 검증 및 보정 모듈
"""
import numpy as np
import pandas as pd

class PredictionValidator:
    def __init__(self, historical_data, constraints=None):
        """
        Args:
            historical_data: 과거 데이터 (통계 계산에 사용)
            constraints: 제약 조건 딕셔너리
        """
        self.historical_data = historical_data
        
        # 기본 제약 조건
        self.constraints = {
            'max_daily_change_pct': 1.5,    # 일일 최대 변화율 (%)
            'max_weekly_change_pct': 5.0,    # 주간 최대 변화율 (%)
            'min_correlation': 0.5,          # 연료간 최소 상관관계
            'historical_volatility_factor': 1.2  # 역사적 변동성 기반 제약
        }
        
        # 사용자 정의 제약 조건으로 업데이트
        if constraints:
            self.constraints.update(constraints)
        
        # 과거 데이터에서 통계 계산
        self._calculate_statistics()
    
    def _calculate_statistics(self):
        """과거 데이터 통계 계산"""
        fuel_cols = ['gasoline', 'premiumGasoline', 'diesel', 'kerosene']
        df = self.historical_data[fuel_cols].copy()
        
        # 일별 변화율 계산
        self.daily_changes = df.pct_change().dropna()
        
        # 주간 변화율 계산 (5일 이동)
        self.weekly_changes = df.pct_change(periods=5).dropna()
        
        # 평균 변화율 및 표준 편차
        self.mean_daily_change = self.daily_changes.mean()
        self.std_daily_change = self.daily_changes.std()
        
        # 연료간 상관관계
        self.fuel_correlation = df.corr()
        
        # 각 연료의 최근 추세 (최근 30일)
        if len(df) >= 30:
            recent = df.iloc[-30:].copy()
            self.recent_trend = recent.apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        else:
            self.recent_trend = pd.Series(0, index=fuel_cols)
    
    def validate_and_adjust(self, predictions_df):
        """
        예측값을 검증하고 필요한 경우 조정
        
        Args:
            predictions_df: 예측 데이터프레임
        
        Returns:
            조정된 예측 데이터프레임
        """
        adjusted_df = predictions_df.copy()
        fuel_cols = adjusted_df.columns
        
        # 1. 일일 변화율 제약
        daily_max_change = self.constraints['max_daily_change_pct'] / 100
        
        for i in range(1, len(adjusted_df)):
            prev_row = adjusted_df.iloc[i-1]
            curr_row = adjusted_df.iloc[i].copy()
            
            for col in fuel_cols:
                change = (curr_row[col] - prev_row[col]) / prev_row[col]
                if abs(change) > daily_max_change:
                    # 방향은 유지하되 변화량을 제한
                    direction = 1 if change > 0 else -1
                    curr_row[col] = prev_row[col] * (1 + direction * daily_max_change)
            
            adjusted_df.iloc[i] = curr_row
        
        # 2. 주간 변화율 제약
        if len(adjusted_df) >= 5:
            weekly_max_change = self.constraints['max_weekly_change_pct'] / 100
            first_row = adjusted_df.iloc[0]
            last_row = adjusted_df.iloc[-1].copy()
            
            for col in fuel_cols:
                weekly_change = (last_row[col] - first_row[col]) / first_row[col]
                if abs(weekly_change) > weekly_max_change:
                    # 주간 변화율 제한 (마지막 날의 값 조정)
                    direction = 1 if weekly_change > 0 else -1
                    last_row[col] = first_row[col] * (1 + direction * weekly_max_change)
            
            adjusted_df.iloc[-1] = last_row
        
        # 3. 연료간 상관관계 제약
        # 최근 상관관계에 따라 조정
        actual_changes = adjusted_df.pct_change().dropna()
        
        # 가솔린을 기준으로 다른 연료와의 방향성 조정
        if 'gasoline' in fuel_cols and len(fuel_cols) > 1:
            ref_changes = actual_changes['gasoline'].values
            
            for col in fuel_cols:
                if col != 'gasoline':
                    # 과거 상관관계 확인
                    hist_corr = self.fuel_correlation.loc['gasoline', col]
                    
                    # 예측된 상관관계가 과거 상관관계와 매우 다른 경우 조정
                    if hist_corr > self.constraints['min_correlation']:
                        for i in range(1, len(adjusted_df)):
                            # 기준 연료와 방향이 다르고 상관관계가 높아야 하는 경우
                            gas_change = ref_changes[i-1] if i-1 < len(ref_changes) else 0
                            
                            # 변화 방향이 반대일 경우, 덜 급격하게 조정
                            if gas_change * (adjusted_df.iloc[i, adjusted_df.columns.get_loc(col)] - 
                                             adjusted_df.iloc[i-1, adjusted_df.columns.get_loc(col)]) < 0:
                                # 방향을 동일하게 조정하되 원래 값의 영향도 유지
                                adjusted_df.iloc[i, adjusted_df.columns.get_loc(col)] = \
                                    adjusted_df.iloc[i-1, adjusted_df.columns.get_loc(col)] * \
                                    (1 + 0.5 * hist_corr * gas_change)
        
        return adjusted_df

# 예측 검증 사용 예:
"""
# 예측값 검증 및 조정 코드
historical_data = df.copy()  # 과거 데이터
validator = PredictionValidator(historical_data)
predictions_df = ...  # 예측값
adjusted_predictions = validator.validate_and_adjust(predictions_df)
"""
