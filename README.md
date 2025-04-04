# 🛒 고객 구매 행동 예측 모델

## 📌 프로젝트 개요
유저 로그 데이터를 기반으로 고객의 구매 가능성을 예측하는 머신러닝 모델을 구축했습니다. 마케팅 전략에 실질적인 인사이트를 제공하여 **전환율 11% 개선**을 달성했습니다.

## 🧰 기술 스택
- **Pandas**: 데이터 수집, 정제 및 feature engineering
- **Scikit-learn**: 데이터 분할, 모델 평가, 베이스라인 구축
- **LightGBM**: 최종 예측 모델, 고속 학습 및 고성능 달성
- **SHAP**: 모델 해석 및 feature importance 시각화

## 👤 담당 역할
- **데이터 라벨링 자동화**  
  - 유저 행동 로그 기반 이탈/구매 여부 자동 태깅 스크립트 작성
- **모델 성능 개선 및 튜닝**  
  - 하이퍼파라미터 튜닝, 클래스 불균형 처리(SMOTE, class_weight 조정)
- **실시간 테스트 환경 구현**  
  - 샘플 로그를 실시간 입력받아 예측값 반환하는 테스트 파이프라인 구축

## 📊 주요 성과
- **F1 Score**: `0.81` (Precision과 Recall의 균형 최적화)
- **Feature Importance Report (SHAP)** 작성  
  - 주요 전환 요인 파악 (예: 페이지 체류 시간, 재방문 빈도 등)
- **마케팅 팀 협업으로 전환율 11% 상승 유도**  
  - 캠페인 대상 고객군 선별 → 효율적 마케팅 실행 가능

## 🔍 핵심 포인트
- **Explainable ML**을 통해 비즈니스 팀과 소통 가능한 인사이트 제공
- 단순 예측 모델을 넘어 **실시간 테스트 환경**까지 고려한 엔드 투 엔드 설계
- 마케팅 전략 수립에 직접 기여한 **실질적 가치 창출**

## 📂 프로젝트 구조
```
customer-purchase-prediction/
├── data/
│   ├── raw/                 # 원본 데이터
│   └── processed/           # 전처리된 데이터
├── notebooks/               # Jupyter 노트북
├── src/
│   ├── data/               # 데이터 처리 관련 코드
│   ├── features/           # 특성 엔지니어링
│   ├── models/             # 모델 관련 코드
│   └── visualization/      # 시각화 관련 코드
├── tests/                  # 테스트 코드
├── requirements.txt        # 프로젝트 의존성
└── README.md              # 프로젝트 문서
```

## 🚀 시작하기
1. 저장소 클론
```bash
git clone [repository-url]
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

4. Jupyter 노트북 실행
```bash
jupyter notebook
```

5. Streamlit 앱 실행
```bash
python -m streamlit run app.py
```

## 🌐 Streamlit 배포 링크
이 프로젝트는 Streamlit Cloud에 배포되어 있습니다. 아래 링크를 통해 대시보드를 실시간으로 확인할 수 있습니다:

[구매 예측 분석 대시보드](https://customer-purchase-prediction1.streamlit.app/)

## 📊 대시보드 기능
- **데이터 개요**: 샘플 데이터, 전처리된 데이터, 생성된 특성, 시간대별 페이지 체류 시간 시각화
- **모델 성능**: F1 스코어, 정밀도, 재현율 및 혼동 행렬
- **특성 중요도**: 모델 예측에 영향을 미치는 주요 특성들의 중요도 시각화

## 📝 Streamlit Cloud 배포 방법
1. GitHub에 코드 푸시
2. [Streamlit 공식 홈페이지](https://streamlit.io/) 접속 후 계정 생성
3. "New app" 선택 후 GitHub 저장소, 브랜치, app.py 파일 경로 지정
4. 배포 완료 후 생성된 URL로 접속 가능 
