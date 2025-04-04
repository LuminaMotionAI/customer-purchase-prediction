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
@st.cache_resource(ttl=3600, max_entries=10)
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
    st.header("데이터 탐색")
    
    # 상위 몇 개의 행 보기
    st.subheader("데이터 샘플")
    st.dataframe(processed_data.head())
    
    # 기본 통계량 보기
    st.subheader("기본 통계량")
    st.dataframe(processed_data.describe())
    
    # 데이터 인사이트 추가
    st.subheader("데이터 인사이트")
    st.markdown("""
    📊 **주요 발견:**
    - 이 데이터셋은 고객별 특성과 구매 여부를 포함하고 있습니다.
    - 기본 통계량을 통해 각 특성의 분포를 파악할 수 있습니다.
    - 특성들 간의 편차가 있으므로, 모델링 전에 정규화/표준화 작업이 적용되었습니다.
    
    💡 **활용 방안:**
    - 특성들의 분포를 파악하여 마케팅 타겟팅의 기준점으로 활용할 수 있습니다.
    - 이상치(outlier)가 있는지 확인하고 데이터 품질을 개선할 수 있습니다.
    """)
    
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
    st.header("모델 성능")
    
    # 정확도 계산 추가 (없는 경우)
    # 예측 결과와 실제 값을 사용하여 정확도 계산
    y_pred_binary = (predictions > 0.5).astype(int)
    accuracy = (y_pred_binary == X_test.index.map(processed_data['target'].to_dict())).mean()
    
    # 성능 지표 보여주기
    st.subheader("성능 지표")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("정확도(Accuracy)", f"{accuracy:.2f}")
    with col2:
        st.metric("정밀도(Precision)", f"{metrics['precision']:.2f}")
    with col3:
        st.metric("재현율(Recall)", f"{metrics['recall']:.2f}")
    with col4:
        st.metric("F1 점수", f"{metrics['f1_score']:.2f}")
    
    # 성능 지표 인사이트 추가
    st.markdown("""
    📊 **모델 성능 해석:**
    - **정확도(Accuracy)**: 전체 예측 중 올바른 예측의 비율입니다. 높을수록 좋습니다.
    - **정밀도(Precision)**: 구매로 예측한 고객 중 실제로 구매한 고객의 비율입니다. 마케팅 비용 최적화에 중요합니다.
    - **재현율(Recall)**: 실제 구매 고객 중 모델이 정확히 식별한 비율입니다. 잠재 고객을 놓치지 않는데 중요합니다.
    - **F1 점수**: 정밀도와 재현율의 조화평균으로, 두 지표 간의 균형을 나타냅니다.
    
    💡 **비즈니스 인사이트:**
    - 재현율이 낮다면 실제 구매 가능성이 있는 고객을 놓치고 있을 수 있습니다.
    - 정밀도가 낮다면 마케팅 비용이 효율적으로 사용되지 않고 있을 수 있습니다.
    - 이 모델의 성능을 기반으로 마케팅 전략을 조정할 수 있습니다.
    """)
    
    # 예측 결과 시각화
    st.subheader("실제 값과 예측 값 비교")
    results_df = X_test.copy()
    results_df['actual'] = X_test.index.map(processed_data['target'].to_dict())
    results_df['predicted'] = predictions

    # 혼동 행렬을 계산합니다
    conf_matrix = pd.crosstab(results_df['actual'], results_df['predicted'], 
                              rownames=['Actual'], colnames=['Predicted'], 
                              normalize='index')

    # 대안 1: 혼동 행렬을 표로 표시 (가장 깔끔한 방법)
    st.write("#### 혼동 행렬 (정규화된 %)")
    # 퍼센트로 변환하여 표시
    conf_percent = conf_matrix.mul(100).round(1).astype(str) + '%'
    st.dataframe(conf_percent, use_container_width=True)

    # 대안 2: 히트맵 그리기 (텍스트 없이)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, cmap='Blues', ax=ax, annot=False, cbar=True)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    # 틱 위치 명시적으로 설정
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(['0', '1'])
    ax.set_yticklabels(['0', '1'])
    st.pyplot(fig)

    # 혼동 행렬 해석 추가
    st.write("#### 혼동 행렬 해석:")
    true_negative = conf_matrix.iloc[0, 0] if 0 in conf_matrix.index and 0 in conf_matrix.columns else 0
    false_positive = conf_matrix.iloc[0, 1] if 0 in conf_matrix.index and 1 in conf_matrix.columns else 0
    false_negative = conf_matrix.iloc[1, 0] if 1 in conf_matrix.index and 0 in conf_matrix.columns else 0
    true_positive = conf_matrix.iloc[1, 1] if 1 in conf_matrix.index and 1 in conf_matrix.columns else 0

    col1, col2 = st.columns(2)
    with col1:
        st.metric("정확히 예측한 비구매 (True Negative)", f"{true_negative:.1%}")
        st.metric("잘못 예측한 구매 (False Positive)", f"{false_positive:.1%}")
    with col2:
        st.metric("잘못 예측한 비구매 (False Negative)", f"{false_negative:.1%}")
        st.metric("정확히 예측한 구매 (True Positive)", f"{true_positive:.1%}")
    
    # 혼동 행렬 인사이트 추가
    st.markdown("""
    📊 **혼동 행렬 인사이트:**
    - **True Negative**: 구매하지 않을 고객을 정확히 예측한 비율입니다. 마케팅 비용 절감에 도움이 됩니다.
    - **False Positive**: 구매하지 않을 고객을 구매할 것으로 잘못 예측한 비율입니다. 이는 마케팅 예산 낭비를 의미합니다.
    - **False Negative**: 실제 구매 고객을 놓친 비율입니다. 이는 잠재적 매출 손실을 의미합니다.
    - **True Positive**: 실제 구매 고객을 정확히 예측한 비율입니다. 효과적인 마케팅 타겟팅을 나타냅니다.
    
    💡 **비즈니스 시사점:**
    - False Positive 비율이 높으면 마케팅 예산 낭비가 발생합니다.
    - False Negative 비율이 높으면 매출 기회를 놓치게 됩니다.
    - 이 정보를 바탕으로 마케팅 전략을 최적화하여 ROI를 향상시킬 수 있습니다.
    """)

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
    
    # 특성 중요도 인사이트 추가
    st.subheader("특성 중요도 인사이트")
    
    # 중요 특성 추출 (상위 3개)
    top_features = importance_df.head(3)['Feature'].tolist()
    top_features_str = ', '.join([f"**{f}**" for f in top_features])
    
    st.markdown(f"""
    📊 **주요 발견:**
    - 상위 중요 특성은 {top_features_str}입니다.
    - 이 특성들이 모델의 예측 결과에 가장 큰 영향을 미칩니다.
    
    💡 **비즈니스 활용 방안:**
    - 중요도가 높은 특성에 집중하여 마케팅 메시지를 개발할 수 있습니다.
    - 제품 개발 및 서비스 개선 시 이러한 중요 특성을 고려할 수 있습니다.
    - 고객 세그먼트 정의에 이 특성들을 활용하여 더 정교한 타겟팅이 가능합니다.
    - 데이터 수집 시 이러한 중요 특성에 관련된 정보를 우선적으로 확보할 필요가 있습니다.
    """)

# 푸터
st.markdown("---")
st.caption("© 2025 구매 예측 분석 대시보드 | Created with Forezy") 