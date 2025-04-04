import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.data_processor import DataProcessor
from src.models.model import PurchasePredictor
import os

# 페이지 설정
st.set_page_config(
    page_title="구매 예측 분석 대시보드",
    page_icon="🛍️",
    layout="wide"
)

# 타이틀 및 설명
st.title("구매 예측 분석 대시보드")
st.markdown("""
이 대시보드는 사용자 행동 데이터를 기반으로 구매 가능성을 예측하는 모델의 결과를 시각화합니다.

[구매 예측 분석 대시보드](https://customer-purchase-prediction1.streamlit.app/)
""")

# 사이드바
st.sidebar.title("설정")
sample_size = st.sidebar.slider("샘플 크기", 100, 1000, 500)
random_seed = st.sidebar.number_input("랜덤 시드", 0, 100, 42)

# 데이터 생성 함수
@st.cache_data
def generate_sample_data(n_samples, seed=42):
    np.random.seed(seed)
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='h'),
        'user_id': np.random.randint(1, 100, n_samples),
        'page_id': np.random.randint(1, 50, n_samples),
        'session_id': np.random.randint(1, 200, n_samples),
        'page_duration': np.random.randint(1, 3600, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    })

# 데이터 및 모델 처리
@st.cache_resource
def process_data_and_model(data):
    # 데이터 처리기와 모델 초기화
    data_processor = DataProcessor()
    model = PurchasePredictor()
    
    # 데이터 전처리
    processed_data = data_processor.preprocess_data(data)
    feature_data = data_processor.create_features(processed_data)
    
    # 데이터 분할 및 스케일링
    X_train, X_test, y_train, y_test = data_processor.split_data(feature_data, 'target')
    X_train_scaled, X_test_scaled = data_processor.scale_features(X_train, X_test)
    
    # 모델 학습
    model.train(X_train_scaled, y_train)
    
    # 예측 및 평가
    predictions = model.predict(X_test_scaled)
    metrics = model.evaluate(X_test_scaled, y_test)
    
    # 특성 중요도
    feature_importance = model.get_feature_importance()
    
    return processed_data, feature_data, X_test, predictions, metrics, feature_importance

# 메인 앱 로직
data = generate_sample_data(sample_size, random_seed)

with st.spinner("데이터 처리 및 모델 훈련 중..."):
    processed_data, feature_data, X_test, predictions, metrics, feature_importance = process_data_and_model(data)

# 탭 생성
tab1, tab2, tab3 = st.tabs(["데이터 개요", "모델 성능", "특성 중요도"])

# 탭 1: 데이터 개요
with tab1:
    st.header("샘플 데이터")
    st.dataframe(data.head(10))
    
    st.header("전처리된 데이터")
    st.dataframe(processed_data.head(10))
    
    st.header("생성된 특성")
    st.dataframe(feature_data.head(10))
    
    # 시간대별 페이지 체류 시간 시각화
    st.subheader("시간대별 평균 페이지 체류 시간")
    
    # 원본 데이터로 시간대별 분석 진행 (processed_data 대신 data 사용)
    hourly_data = data.groupby(data['timestamp'].dt.hour)['page_duration'].mean().reset_index()
    hourly_data.columns = ['Hour', 'Average Page Duration']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Hour', y='Average Page Duration', data=hourly_data, ax=ax)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Page Duration (seconds)')
    st.pyplot(fig)

# 탭 2: 모델 성능
with tab2:
    st.header("모델 성능 지표")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("F1 Score", f"{metrics['f1_score']:.4f}")
    col2.metric("Precision", f"{metrics['precision']:.4f}")
    col3.metric("Recall", f"{metrics['recall']:.4f}")
    
    # 예측 결과 시각화
    st.subheader("실제 값과 예측 값 비교")
    results_df = X_test.copy()
    results_df['actual'] = X_test.index.map(processed_data['target'].to_dict())
    results_df['predicted'] = predictions
    
    fig, ax = plt.subplots(figsize=(10, 6))
    conf_matrix = pd.crosstab(results_df['actual'], results_df['predicted'], 
                              rownames=['Actual'], colnames=['Predicted'], 
                              normalize='index')
    sns.heatmap(conf_matrix, annot=True, fmt='.2%', cmap='Blues', ax=ax)
    st.pyplot(fig)

# 탭 3: 특성 중요도
with tab3:
    st.header("특성 중요도")
    
    # 특성 중요도를 데이터프레임으로 변환
    importance_df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    }).sort_values('Importance', ascending=False)
    
    # 바 차트로 시각화
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    st.pyplot(fig)
    
    st.subheader("특성 중요도 데이터")
    st.dataframe(importance_df)

# 푸터
st.markdown("---")
st.caption("© 2025 구매 예측 분석 대시보드 | Created with Forezy") 
