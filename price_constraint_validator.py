"""
유가 예측 결과에 현실적인 제약조건을 적용하는 검증기
"""
import pandas as pd
import numpy as np

class FuelPriceConstraintValidator:
    """유가 예측에 현실적인 제약조건을 적용하는 검증기"""
    
    def __init__(self, max_daily_change_pct=1.0, max_weekly_change_pct=3.5, 
                 smoothing_factor=0.6, price_relationships=None):
        """
        초기화
        
        Args:
            max_daily_change_pct: 일일 최대 변동률(%)
            max_weekly_change_pct: 주간 최대 변동률(%)
            smoothing_factor: 스무딩 계수 (0-1, 클수록 더 매끄러움)
            price_relationships: 유종간 가격 관계 설정 (딕셔너리)
        """
        self.max_daily_change_pct = max_daily_change_pct
        self.max_weekly_change_pct = max_weekly_change_pct
        self.smoothing_factor = smoothing_factor
        
        # 기본 유종간 가격 차이 설정
        self.price_relationships = {
            'premium_gasoline_to_gasoline': 200,  # 고급유-일반유 최소 차이
            'gasoline_to_diesel': 100,            # 일반유-경유 최소 차이
            'diesel_to_kerosene': 50              # 경유-등유 최소 차이
        }
        
        # 사용자 정의 가격 관계 설정이 있으면 업데이트
        if price_relationships:
            self.price_relationships.update(price_relationships)
            
    def validate_predictions(self, predictions_df):
        """
        예측 결과에 제약조건을 적용하여 보정
        
        Args:
            predictions_df: 예측 데이터프레임 (날짜 인덱스, 유종별 컬럼)
        
        Returns:
            보정된 예측 데이터프레임
        """
        # 예측 데이터 복사
        df = predictions_df.copy()
        
        # 1. 일일 가격 변동 제약
        self._apply_daily_changes_constraint(df)
        
        # 2. 유종간 가격 관계 제약
        self._apply_price_relationships(df)
        
        # 3. 주간 총 변동 제약
        self._apply_weekly_change_constraint(df)
        
        # 4. 스무딩 적용
        self._apply_smoothing(df)
        
        return df
    
    def _apply_daily_changes_constraint(self, df):
        """일일 가격 변동 제약 적용"""
        max_change_ratio = self.max_daily_change_pct / 100.0
        
        # 두번째 날부터 적용
        for i in range(1, len(df)):
            prev_row = df.iloc[i-1]
            
            # 각 유종별 변동률 제약
            for col in df.columns:
                curr_val = df.iloc[i][col]
                prev_val = prev_row[col]
                
                if prev_val > 0:  # 0으로 나누기 방지
                    change_pct = (curr_val - prev_val) / prev_val
                    
                    # 최대 변동폭 초과시 조정
                    if abs(change_pct) > max_change_ratio:
                        direction = 1 if change_pct > 0 else -1
                        new_val = prev_val * (1 + direction * max_change_ratio)
                        df.iloc[i, df.columns.get_loc(col)] = new_val
    
    def _apply_price_relationships(self, df):
        """유종간 가격 관계 제약 적용"""
        for i in range(len(df)):
            # 고급휘발유 > 일반휘발유
            if 'premiumGasoline' in df.columns and 'gasoline' in df.columns:
                min_diff = self.price_relationships['premium_gasoline_to_gasoline']
                if df.iloc[i]['premiumGasoline'] < df.iloc[i]['gasoline'] + min_diff:
                    df.iloc[i, df.columns.get_loc('premiumGasoline')] = df.iloc[i]['gasoline'] + min_diff
            
            # 일반휘발유 > 경유
            if 'gasoline' in df.columns and 'diesel' in df.columns:
                min_diff = self.price_relationships['gasoline_to_diesel']
                if df.iloc[i]['gasoline'] < df.iloc[i]['diesel'] + min_diff:
                    df.iloc[i, df.columns.get_loc('gasoline')] = df.iloc[i]['diesel'] + min_diff
            
            # 경유 > 등유
            if 'diesel' in df.columns and 'kerosene' in df.columns:
                min_diff = self.price_relationships['diesel_to_kerosene']
                if df.iloc[i]['diesel'] < df.iloc[i]['kerosene'] + min_diff:
                    df.iloc[i, df.columns.get_loc('diesel')] = df.iloc[i]['kerosene'] + min_diff
    
    def _apply_weekly_change_constraint(self, df):
        """주간 총 변동 제약 적용"""
        if len(df) <= 1:
            return
            
        max_weekly_change = self.max_weekly_change_pct / 100.0
        
        # 첫날과 마지막날 비교
        first_day = df.iloc[0]
        last_day = df.iloc[-1].copy()
        
        # 각 유종별 주간 변동폭 제약
        for col in df.columns:
            if first_day[col] > 0:  # 0으로 나누기 방지
                total_change_pct = (last_day[col] - first_day[col]) / first_day[col]
                
                # 최대 변동폭 초과시 마지막 값 조정
                if abs(total_change_pct) > max_weekly_change:
                    direction = 1 if total_change_pct > 0 else -1
                    last_day[col] = first_day[col] * (1 + direction * max_weekly_change)
        
        # 조정된 마지막날 값 적용
        df.iloc[-1] = last_day
    
    def _apply_smoothing(self, df):
        """시계열 스무딩 적용 - 급격한 변화 완화"""
        # 스무딩 강도가 0이면 적용하지 않음
        if self.smoothing_factor == 0:
            return
            
        # 원본 값 백업
        original_df = df.copy()
        
        # 두번째 날부터 스무딩 적용
        for i in range(1, len(df)):
            for col in df.columns:
                # 이전 값과 현재 예측값의 가중 평균 계산
                df.iloc[i, df.columns.get_loc(col)] = (
                    self.smoothing_factor * df.iloc[i-1][col] + 
                    (1 - self.smoothing_factor) * original_df.iloc[i][col]
                )
