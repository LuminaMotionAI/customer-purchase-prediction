import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd

class Visualizer:
    def __init__(self):
        plt.style.use('seaborn')
        
    def plot_feature_importance(self, feature_importance, top_n=10):
        """특성 중요도 시각화"""
        plt.figure(figsize=(10, 6))
        
        # 상위 N개 특성 추출
        top_features = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n])
        
        # 막대 그래프 생성
        plt.bar(
            range(len(top_features)),
            list(top_features.values()),
            align='center'
        )
        
        # x축 레이블 설정
        plt.xticks(
            range(len(top_features)),
            list(top_features.keys()),
            rotation=45,
            ha='right'
        )
        
        plt.title('Top Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_shap_values(self, model, X):
        """SHAP 값 시각화"""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # SHAP 요약 플롯
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            plot_type="bar",
            show=False
        )
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_learning_curve(self, history):
        """학습 곡선 시각화"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(history['train_loss'], label='Training Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss')
            
        plt.title('Learning Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        return plt.gcf()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """혼동 행렬 시각화"""
        plt.figure(figsize=(8, 6))
        cm = pd.crosstab(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        return plt.gcf() 