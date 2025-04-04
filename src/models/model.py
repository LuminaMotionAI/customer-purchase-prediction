import lightgbm as lgb
from sklearn.metrics import f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import numpy as np

class PurchasePredictor:
    def __init__(self):
        self.model = None
        self.feature_importance = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """모델 학습"""
        # SMOTE를 사용한 클래스 불균형 처리
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # LightGBM 데이터셋 생성
        train_data = lgb.Dataset(X_train_balanced, label=y_train_balanced)
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        else:
            valid_data = None
            
        # 하이퍼파라미터 설정
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # 모델 학습
        callbacks = []
        if valid_data:
            callbacks.append(lgb.early_stopping(stopping_rounds=50))
        callbacks.append(lgb.log_evaluation(period=100))
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data] if valid_data else None,
            callbacks=callbacks
        )
        
        # 특성 중요도 저장
        self.feature_importance = dict(zip(
            self.model.feature_name(),
            self.model.feature_importance()
        ))
        
    def predict(self, X):
        """예측 수행"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """모델 평가"""
        y_pred = self.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        metrics = {
            'f1_score': f1_score(y_test, y_pred_binary),
            'precision': precision_score(y_test, y_pred_binary),
            'recall': recall_score(y_test, y_pred_binary)
        }
        
        return metrics
    
    def get_feature_importance(self):
        """특성 중요도 반환"""
        return self.feature_importance 