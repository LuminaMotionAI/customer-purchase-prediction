import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_and_preprocess_data():
    """데이터 로드 및 전처리"""
    try:
        df = pd.read_csv('data/raw/coffee_shop_revenue.csv')
        return df
    except Exception as e:
        logging.error(f"데이터 로드 중 오류 발생: {str(e)}")
        raise

def analyze_basic_stats(df):
    """기본 통계 분석"""
    logging.info("\n=== 기본 통계 분석 ===")
    logging.info(f"총 데이터 수: {len(df)}")
    logging.info("\n매출 통계:")
    logging.info(df['Daily_Revenue'].describe())
    
    # 결측치 확인
    missing_values = df.isnull().sum()
    if missing_values.any():
        logging.info("\n결측치 정보:")
        logging.info(missing_values[missing_values > 0])

def analyze_correlations(df):
    """상관관계 분석"""
    logging.info("\n=== 상관관계 분석 ===")
    
    # 상관관계 히트맵
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('변수 간 상관관계')
    plt.tight_layout()
    plt.savefig('data/processed/correlation_heatmap.png')
    plt.close()
    
    # 매출과의 상관관계
    revenue_corr = df.corr()['Daily_Revenue'].sort_values(ascending=False)
    logging.info("\n매출과의 상관관계:")
    logging.info(revenue_corr)

def analyze_feature_distributions(df):
    """특성 분포 분석"""
    logging.info("\n=== 특성 분포 분석 ===")
    
    # 각 특성의 분포 시각화
    for column in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=column, bins=30)
        plt.title(f'{column} 분포')
        plt.tight_layout()
        plt.savefig(f'data/processed/{column}_distribution.png')
        plt.close()

def analyze_relationships(df):
    """매출과 주요 변수 간의 관계 분석"""
    logging.info("\n=== 매출과 주요 변수 간의 관계 분석 ===")
    
    # 주요 변수와 매출의 산점도
    key_features = ['Number_of_Customers_Per_Day', 'Average_Order_Value', 
                   'Operating_Hours_Per_Day', 'Marketing_Spend_Per_Day']
    
    for feature in key_features:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=feature, y='Daily_Revenue')
        plt.title(f'{feature}와 매출의 관계')
        plt.tight_layout()
        plt.savefig(f'data/processed/{feature}_vs_revenue.png')
        plt.close()

def main():
    try:
        # 데이터 로드
        df = load_and_preprocess_data()
        
        # 분석 실행
        analyze_basic_stats(df)
        analyze_correlations(df)
        analyze_feature_distributions(df)
        analyze_relationships(df)
        
        logging.info("\n분석이 완료되었습니다. 결과는 data/processed 디렉토리에서 확인할 수 있습니다.")
        
    except Exception as e:
        logging.error(f"분석 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 