import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """데이터 로드"""
        return pd.read_csv(file_path)
    
    def preprocess_data(self, df):
        """기본 전처리 수행"""
        # 결측치 처리
        df = df.fillna(0)
        
        # 시간 관련 특성 변환
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df = df.drop('timestamp', axis=1)  # timestamp 컬럼 제거
            
        return df
    
    def create_features(self, df):
        """특성 엔지니어링"""
        # 페이지 체류 시간 관련 특성
        if 'page_duration' in df.columns:
            df['avg_page_duration'] = df.groupby('user_id')['page_duration'].transform('mean')
            df['max_page_duration'] = df.groupby('user_id')['page_duration'].transform('max')
            
        # 방문 빈도 관련 특성
        if 'visit_count' in df.columns:
            df['visit_frequency'] = df.groupby('user_id')['visit_count'].transform('count')
            
        return df
    
    def split_data(self, df, target_col):
        """데이터 분할"""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """특성 스케일링"""
        # DataFrame을 numpy array로 변환
        X_train_array = X_train.values
        X_test_array = X_test.values
        
        X_train_scaled = self.scaler.fit_transform(X_train_array)
        X_test_scaled = self.scaler.transform(X_test_array)
        
        # 스케일링된 array를 다시 DataFrame으로 변환
        X_train_scaled = pd.DataFrame(
            X_train_scaled,
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled,
            columns=X_test.columns,
            index=X_test.index
        )
        
        return X_train_scaled, X_test_scaled 