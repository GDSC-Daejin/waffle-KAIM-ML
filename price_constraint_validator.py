"""
유가 예측 결과에 현실적인 제약조건을 적용하는 검증기
"""
import pandas as pd
import numpy as np

class FuelPriceConstraintValidator:
    """유가 예측에 현실적인 제약조건을 적용하는 검증기"""
    
    def __init__(self, max_daily_change_pct=1.0, max_weekly_change_pct=3.5, 
                 smoothing_factor=0.6, historical_data=None, current_prices=None,
                 price_relationships=None):
        """
        초기화
        
        Args:
            max_daily_change_pct: 일일 최대 변동률(%)
            max_weekly_change_pct: 주간 최대 변동률(%)
            smoothing_factor: 스무딩 계수 (0-1, 클수록 더 매끄러움)
            historical_data: 과거 데이터 (유종 간 가격 관계 학습용)
            current_prices: 현재 시장 가격 (딕셔너리) - 앵커링용
            price_relationships: 유종간 가격 관계 설정 (딕셔너리)
        """
        self.max_daily_change_pct = max_daily_change_pct
        self.max_weekly_change_pct = max_weekly_change_pct
        self.smoothing_factor = smoothing_factor
        
        # 현재 시장 가격 설정 (앵커링용)
        self.current_prices = {
            'gasoline': 1712.37,
            'premiumGasoline': 1949.42,
            'diesel': 1578.28,
            'kerosene': 1338.96
        }
        
        # 사용자 정의 현재 가격이 있으면 업데이트
        if current_prices:
            self.current_prices.update(current_prices)
        
        # 유종 간 가격 차이의 통계적 특성 계산
        self.price_stats = self._calculate_price_relationship_stats(historical_data)
        
        # 명시적으로 설정된 가격 관계 처리
        if price_relationships:
            # min_diff 값에 직접 매핑
            if 'premium_gasoline_to_gasoline' in price_relationships:
                self.price_stats['min_diff']['premium_to_gasoline'] = price_relationships['premium_gasoline_to_gasoline']
            
            if 'gasoline_to_diesel' in price_relationships:
                self.price_stats['min_diff']['gasoline_to_diesel'] = price_relationships['gasoline_to_diesel']
            
            if 'diesel_to_kerosene' in price_relationships:
                self.price_stats['min_diff']['diesel_to_kerosene'] = price_relationships['diesel_to_kerosene']
            
    def _calculate_price_relationship_stats(self, historical_data):
        """과거 데이터에서 유종간 가격 관계 통계 계산"""
        stats = {
            # 기본 평균 가격 차이 (현재 시장 가격 기반)
            'avg_diff': {
                'premium_to_gasoline': self.current_prices['premiumGasoline'] - self.current_prices['gasoline'],
                'gasoline_to_diesel': self.current_prices['gasoline'] - self.current_prices['diesel'],
                'diesel_to_kerosene': self.current_prices['diesel'] - self.current_prices['kerosene']
            },
            # 기본 변동 범위 (현실적인 값)
            'std_diff': {
                'premium_to_gasoline': 30.0,  # 약 ±30원 변동 가능
                'gasoline_to_diesel': 25.0,   # 약 ±25원 변동 가능
                'diesel_to_kerosene': 20.0    # 약 ±20원 변동 가능
            },
            # 최소 가격 차이 (이보다 작으면 비정상)
            'min_diff': {
                'premium_to_gasoline': 150.0,  # 최소 150원 차이
                'gasoline_to_diesel': 80.0,    # 최소 80원 차이
                'diesel_to_kerosene': 20.0     # 최소 20원 차이
            },
            # 최대 가격 차이 (이보다 크면 비정상)
            'max_diff': {
                'premium_to_gasoline': 300.0,  # 최대 300원 차이
                'gasoline_to_diesel': 200.0,   # 최대 200원 차이
                'diesel_to_kerosene': 150.0    # 최대 150원 차이
            }
        }
        
        # 과거 데이터가 있으면 통계 업데이트
        if historical_data is not None and isinstance(historical_data, pd.DataFrame):
            try:
                # 필요한 컬럼이 모두 있는지 확인
                req_cols = ['gasoline', 'premiumGasoline', 'diesel', 'kerosene']
                if all(col in historical_data.columns for col in req_cols):
                    # 최근 3개월 데이터만 사용 (최신 경향 반영)
                    recent_data = historical_data.iloc[-90:] if len(historical_data) > 90 else historical_data
                    
                    # 평균 가격 차이 계산
                    stats['avg_diff']['premium_to_gasoline'] = (
                        recent_data['premiumGasoline'] - recent_data['gasoline']
                    ).mean()
                    
                    stats['avg_diff']['gasoline_to_diesel'] = (
                        recent_data['gasoline'] - recent_data['diesel']
                    ).mean()
                    
                    stats['avg_diff']['diesel_to_kerosene'] = (
                        recent_data['diesel'] - recent_data['kerosene']
                    ).mean()
                    
                    # 표준 편차 계산 (변동성)
                    stats['std_diff']['premium_to_gasoline'] = (
                        recent_data['premiumGasoline'] - recent_data['gasoline']
                    ).std()
                    
                    stats['std_diff']['gasoline_to_diesel'] = (
                        recent_data['gasoline'] - recent_data['diesel']
                    ).std()
                    
                    stats['std_diff']['diesel_to_kerosene'] = (
                        recent_data['diesel'] - recent_data['kerosene']
                    ).std()
                    
                    # 최소/최대 차이 (평균의 ±2σ 범위로 설정)
                    for rel in ['premium_to_gasoline', 'gasoline_to_diesel', 'diesel_to_kerosene']:
                        avg = stats['avg_diff'][rel]
                        std = stats['std_diff'][rel]
                        stats['min_diff'][rel] = max(10.0, avg - 2 * std)  # 최소 10원 보장
                        stats['max_diff'][rel] = avg + 2 * std
            except Exception as e:
                print(f"과거 데이터로 가격 관계 통계 계산 중 오류: {e}")
        
        return stats
            
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
        
        # 0. 현재 시장 가격을 참조하여 초기값 앵커링
        self._apply_market_anchoring(df)
        
        # 1. 일일 가격 변동 제약
        self._apply_daily_changes_constraint(df)
        
        # 2. 유종간 가격 관계 제약 (자연스러운 관계 유지)
        self._apply_natural_price_relationships(df)
        
        # 3. 주간 총 변동 제약
        self._apply_weekly_change_constraint(df)
        
        # 4. 스무딩 적용
        self._apply_smoothing(df)
        
        return df
    
    def _apply_market_anchoring(self, df):
        """현재 시장 가격을 첫날 예측에 반영하여 앵커링"""
        if len(df) == 0:
            return
            
        # 첫 날 데이터를 현재 시장 가격에서 최대 1% 이내로 조정
        for col in df.columns:
            if col in self.current_prices:
                market_price = self.current_prices[col]
                predicted_price = df.iloc[0][col]
                
                # 현재 가격과 예측 가격의 차이가 1% 이상인 경우
                if abs(predicted_price - market_price) / market_price > 0.01:
                    # 현재 가격을 중심으로 최대 1% 변동 범위 내에서 조정
                    max_change = market_price * 0.01
                    direction = 1 if predicted_price > market_price else -1
                    new_price = market_price + (direction * max_change)
                    df.iloc[0, df.columns.get_loc(col)] = new_price
    
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
    
    def _apply_natural_price_relationships(self, df):
        """유종간 자연스러운 가격 관계 제약 적용"""
        for i in range(len(df)):
            # 현재 행의 값
            row = df.iloc[i]
            
            # 각 유종 쌍에 대해 처리
            relations = [
                {'high': 'premiumGasoline', 'low': 'gasoline', 'rel': 'premium_to_gasoline'},
                {'high': 'gasoline', 'low': 'diesel', 'rel': 'gasoline_to_diesel'},
                {'high': 'diesel', 'low': 'kerosene', 'rel': 'diesel_to_kerosene'}
            ]
            
            for rel in relations:
                high_price_col = rel['high']
                low_price_col = rel['low']
                rel_name = rel['rel']
                
                # 필요한 컬럼이 존재하는지 확인
                if high_price_col not in df.columns or low_price_col not in df.columns:
                    continue
                
                high_price = row[high_price_col]
                low_price = row[low_price_col]
                price_diff = high_price - low_price
                
                # 통계값 가져오기
                avg_diff = self.price_stats['avg_diff'][rel_name]
                min_diff = self.price_stats['min_diff'][rel_name]
                max_diff = self.price_stats['max_diff'][rel_name]
                
                # 1. 가격 역전 현상 수정 (고급유가 일반유보다 저렴한 경우 등)
                if price_diff < 0:
                    # 가격 관계 복원 (평균 차이의 70%로 설정)
                    target_diff = max(10.0, avg_diff * 0.7)
                    
                    # 두 가격을 모두 약간씩 조정하여 관계 복원
                    adj_high = low_price + target_diff
                    df.iloc[i, df.columns.get_loc(high_price_col)] = adj_high
                
                # 2. 가격 차이가 비정상적으로 작은 경우
                elif price_diff < min_diff:
                    # 최소 차이의 80%만 적용 (완전히 고정된 값으로 보정하지 않음)
                    target_diff = min_diff * 0.8 + price_diff * 0.2
                    
                    # 두 가격 모두 약간씩 조정 (자연스러운 변화)
                    adj_factor = 0.6  # 높은 가격을 60%, 낮은 가격을 40% 조정
                    adj_high = low_price + target_diff
                    adj_low = high_price - target_diff
                    
                    # 급격한 변화가 되지 않도록 원래 가격과 조정된 가격 사이에서 가중 평균
                    new_high = high_price * (1 - adj_factor) + adj_high * adj_factor
                    new_low = low_price * (1 - (1 - adj_factor)) + adj_low * (1 - adj_factor)
                    
                    df.iloc[i, df.columns.get_loc(high_price_col)] = new_high
                    df.iloc[i, df.columns.get_loc(low_price_col)] = new_low
                
                # 3. 가격 차이가 비정상적으로 큰 경우
                elif price_diff > max_diff:
                    # 최대 차이의 80%만 적용
                    target_diff = max_diff * 0.8 + price_diff * 0.2
                    
                    # 두 가격 모두 약간씩 조정 (자연스러운 변화)
                    adj_factor = 0.6
                    adj_high = low_price + target_diff
                    adj_low = high_price - target_diff
                    
                    new_high = high_price * (1 - adj_factor) + adj_high * adj_factor
                    new_low = low_price * (1 - (1 - adj_factor)) + adj_low * (1 - adj_factor)
                    
                    df.iloc[i, df.columns.get_loc(high_price_col)] = new_high
                    df.iloc[i, df.columns.get_loc(low_price_col)] = new_low
    
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
        
        # 마지막 날짜를 기준으로 역방향으로 보간
        if len(df) > 2:
            for col in df.columns:
                start_val = df.iloc[0][col]
                end_val = df.iloc[-1][col]
                
                # 중간값들은 시작과 끝 사이의 선형 보간으로 자연스럽게 조정
                # (기존 예측의 추세는 유지하면서 마지막 값까지 매끄럽게 연결)
                original_vals = df[col].values[1:-1]  # 첫날, 마지막날 제외한 원본값
                
                for i in range(1, len(df) - 1):
                    progress = i / (len(df) - 1)  # 0~1 사이의 진행 비율
                    target = start_val + progress * (end_val - start_val)  # 목표값
                    
                    # 원래 예측값과 목표값을 혼합 (추세 보존)
                    orig_val = df.iloc[i][col]
                    weight = 0.3  # 원본:목표 = 7:3 비율로 혼합
                    df.iloc[i, df.columns.get_loc(col)] = orig_val * (1 - weight) + target * weight
    
    def _apply_smoothing(self, df):
        """시계열 스무딩 적용 - 급격한 변화 완화"""
        # 스무딩 강도가 0이면 적용하지 않음
        if self.smoothing_factor == 0 or len(df) <= 1:
            return
            
        # 원본 값 백업
        original_df = df.copy()
        
        # 이동 평균 스무딩 적용
        alpha = self.smoothing_factor
        
        # 두번째 날부터 스무딩 적용
        for i in range(1, len(df)):
            for col in df.columns:
                # 이전 값과 현재 예측값의 가중 평균 계산
                df.iloc[i, df.columns.get_loc(col)] = (
                    alpha * df.iloc[i-1][col] + 
                    (1 - alpha) * original_df.iloc[i][col]
                )
