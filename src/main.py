import pandas as pd
import numpy as np
from data.data_processor import DataProcessor
from models.model import PurchasePredictor
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    try:
        # 데이터 처리기와 모델 초기화
        data_processor = DataProcessor()
        model = PurchasePredictor()
        
        # 샘플 데이터 생성 (실제 데이터가 없는 경우를 위한 임시 데이터)
        n_samples = 1000
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='H'),
            'user_id': np.random.randint(1, 100, n_samples),
            'page_id': np.random.randint(1, 50, n_samples),
            'session_id': np.random.randint(1, 200, n_samples),
            'page_duration': np.random.randint(1, 3600, n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
        
        # 데이터 전처리
        processed_data = data_processor.preprocess_data(data)
        feature_data = data_processor.create_features(processed_data)
        
        # 데이터 분할 및 스케일링
        X_train, X_test, y_train, y_test = data_processor.split_data(feature_data, 'target')
        X_train_scaled, X_test_scaled = data_processor.scale_features(X_train, X_test)
        
        # 모델 학습
        logging.info("모델 학습 시작...")
        model.train(X_train_scaled, y_train)
        
        # 예측 및 평가
        logging.info("예측 및 평가 시작...")
        predictions = model.predict(X_test_scaled)
        metrics = model.evaluate(X_test_scaled, y_test)
        
        # 결과 출력
        logging.info("모델 성능 평가 결과:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
        # 특성 중요도 확인
        feature_importance = model.get_feature_importance()
        logging.info("\n특성 중요도:")
        for feature, importance in feature_importance.items():
            logging.info(f"{feature}: {importance:.4f}")
            
    except Exception as e:
        logging.error(f"오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 