#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型评估脚本：对训练好的模型进行全面评估。
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import joblib
import shap
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / 'data' / 'processed' / 'features'
MODELS_DIR = ROOT_DIR / 'models'
EVALUATION_DIR = ROOT_DIR / 'evaluation'


def load_model():
    """加载最新训练的模型"""
    logger.info("加载最新模型...")
    
    model_path = MODELS_DIR / "latest_model.joblib"
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    
    model = joblib.load(model_path)
    logger.info("模型加载成功")
    return model


def load_data():
    """加载特征数据"""
    logger.info("加载特征数据...")
    
    filepath = DATA_DIR / 'final_features.csv'
    if not filepath.exists():
        logger.error(f"特征数据文件不存在: {filepath}")
        raise FileNotFoundError(f"找不到特征数据文件: {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"成功加载特征数据，包含 {len(df)} 条记录和 {df.shape[1]} 个特征")
    return df


def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }


def evaluate_basic_metrics(model, X, y):
    """计算基本评估指标"""
    logger.info("计算基本评估指标...")
    
    # 预测
    y_pred = model.predict(X)
    
    # 计算指标
    metrics = calculate_metrics(y, y_pred)
    
    # 输出结果
    logger.info(f"平均绝对误差 (MAE): {metrics['mae']:.4f}")
    logger.info(f"均方误差 (MSE): {metrics['mse']:.4f}")
    logger.info(f"均方根误差 (RMSE): {metrics['rmse']:.4f}")
    logger.info(f"决定系数 (R²): {metrics['r2']:.4f}")
    
    # 计算预测误差
    errors = y - y_pred
    metrics['errors'] = errors
    
    return metrics, y_pred


def evaluate_cross_validation(model, X, y, cv=5):
    """使用交叉验证评估模型"""
    logger.info(f"使用 {cv} 折交叉验证评估模型...")
    
    # 定义评估指标
    scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
    
    # 创建KFold对象
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # 执行交叉验证
    cv_results = {}
    for score in scoring:
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring=score, n_jobs=-1)
        
        # 转换分数
        if 'neg_' in score:
            cv_scores = -cv_scores
            score = score.replace('neg_', '')
        
        cv_results[score] = {
            'scores': cv_scores,
            'mean': np.mean(cv_scores),
            'std': np.std(cv_scores)
        }
    
    # 输出结果
    logger.info("交叉验证结果:")
    for score, result in cv_results.items():
        logger.info(f"{score}: {result['mean']:.4f} ± {result['std']:.4f}")
    
    return cv_results


def analyze_residuals(y_true, y_pred):
    """分析残差"""
    logger.info("分析残差...")
    
    # 计算残差
    residuals = y_true - y_pred
    
    # 计算残差统计信息
    residual_stats = {
        'min': np.min(residuals),
        'max': np.max(residuals),
        'mean': np.mean(residuals),
        'median': np.median(residuals),
        'std': np.std(residuals)
    }
    
    # 输出统计信息
    logger.info(f"残差最小值: {residual_stats['min']:.4f}")
    logger.info(f"残差最大值: {residual_stats['max']:.4f}")
    logger.info(f"残差均值: {residual_stats['mean']:.4f}")
    logger.info(f"残差中位数: {residual_stats['median']:.4f}")
    logger.info(f"残差标准差: {residual_stats['std']:.4f}")
    
    return residuals, residual_stats


def create_evaluation_visualizations(y_true, y_pred, residuals):
    """创建评估可视化"""
    logger.info("创建评估可视化...")
    
    # 创建评估目录
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 实际值与预测值的散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('实际值 vs 预测值')
    plt.grid(True)
    plt.tight_layout()
    scatter_path = EVALUATION_DIR / f"actual_vs_predicted_{timestamp}.png"
    plt.savefig(scatter_path)
    plt.close()
    
    # 2. 残差直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('残差')
    plt.ylabel('频率')
    plt.title('残差分布')
    plt.grid(True)
    plt.tight_layout()
    hist_path = EVALUATION_DIR / f"residual_histogram_{timestamp}.png"
    plt.savefig(hist_path)
    plt.close()
    
    # 3. 残差 vs 预测值
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差 vs 预测值')
    plt.grid(True)
    plt.tight_layout()
    residuals_path = EVALUATION_DIR / f"residuals_vs_predicted_{timestamp}.png"
    plt.savefig(residuals_path)
    plt.close()
    
    # 4. Q-Q图
    plt.figure(figsize=(10, 6))
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('残差Q-Q图')
    plt.grid(True)
    plt.tight_layout()
    qq_path = EVALUATION_DIR / f"qq_plot_{timestamp}.png"
    plt.savefig(qq_path)
    plt.close()
    
    logger.info("评估可视化已保存")
    
    # 返回保存的文件路径
    return {
        'scatter_plot': scatter_path,
        'residual_histogram': hist_path,
        'residuals_vs_predicted': residuals_path,
        'qq_plot': qq_path
    }


def analyze_feature_impact(model, X):
    """分析特征影响（使用SHAP值）"""
    logger.info("分析特征影响（SHAP值）...")
    
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)
    
    # 计算SHAP值（使用样本的子集以提高效率）
    sample_size = min(1000, X.shape[0])
    X_sample = X.sample(sample_size, random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    # 创建评估目录
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # SHAP摘要图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title('SHAP特征重要性摘要')
    plt.tight_layout()
    shap_summary_path = EVALUATION_DIR / f"shap_summary_{timestamp}.png"
    plt.savefig(shap_summary_path)
    plt.close()
    
    # SHAP依赖图 - 对前3个最重要的特征
    feature_importance = np.abs(shap_values).mean(0)
    top_features_idx = feature_importance.argsort()[-3:]
    top_features = X.columns[top_features_idx]
    
    shap_dependency_paths = []
    for feature in top_features:
        plt.figure(figsize=(12, 8))
        shap.dependence_plot(feature, shap_values, X_sample, show=False)
        plt.title(f'SHAP依赖图 - {feature}')
        plt.tight_layout()
        dependency_path = EVALUATION_DIR / f"shap_dependency_{feature}_{timestamp}.png"
        plt.savefig(dependency_path)
        plt.close()
        shap_dependency_paths.append(dependency_path)
    
    logger.info("SHAP分析可视化已保存")
    
    # 返回SHAP值和可视化路径
    return {
        'shap_values': shap_values,
        'shap_summary_path': shap_summary_path,
        'shap_dependency_paths': shap_dependency_paths
    }


def analyze_error_distribution(y_true, y_pred, X):
    """分析错误分布"""
    logger.info("分析错误分布...")
    
    # 计算绝对误差
    abs_errors = np.abs(y_true - y_pred)
    
    # 创建包含误差的DataFrame
    error_df = pd.DataFrame({
        'absolute_error': abs_errors,
        'true_value': y_true,
        'predicted_value': y_pred
    })
    
    # 添加特征列
    for col in X.columns:
        error_df[col] = X[col].values
    
    # 寻找误差最大的样本
    n_worst = 10
    worst_samples = error_df.nlargest(n_worst, 'absolute_error')
    
    # 输出最差预测
    logger.info(f"误差最大的 {n_worst} 个样本:")
    for i, (idx, row) in enumerate(worst_samples.iterrows()):
        logger.info(f"样本 {idx}: 实际值 = {row['true_value']:.2f}, "
                   f"预测值 = {row['predicted_value']:.2f}, "
                   f"绝对误差 = {row['absolute_error']:.2f}")
    
    # 分析误差与特征的关系
    correlations = {}
    for col in X.columns:
        corr = np.corrcoef(error_df['absolute_error'], error_df[col])[0, 1]
        correlations[col] = corr
    
    # 找出与误差相关性最高的特征
    corr_df = pd.DataFrame({
        'feature': list(correlations.keys()),
        'correlation_with_error': list(correlations.values())
    }).sort_values('correlation_with_error', key=abs, ascending=False)
    
    # 输出与误差最相关的特征
    logger.info("与误差最相关的特征:")
    for i, row in corr_df.head(10).iterrows():
        logger.info(f"{row['feature']}: {row['correlation_with_error']:.4f}")
    
    # 创建评估目录
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 将结果保存到文件
    worst_samples_path = EVALUATION_DIR / f"worst_predictions_{timestamp}.csv"
    worst_samples.to_csv(worst_samples_path, index=True)
    
    correlations_path = EVALUATION_DIR / f"error_correlations_{timestamp}.csv"
    corr_df.to_csv(correlations_path, index=False)
    
    logger.info(f"误差分析结果已保存至: {worst_samples_path} 和 {correlations_path}")
    
    return {
        'worst_samples': worst_samples,
        'error_correlations': corr_df
    }


def main():
    """主函数：执行模型评估流程"""
    logger.info("开始模型评估流程")
    
    try:
        # 创建评估目录
        os.makedirs(EVALUATION_DIR, exist_ok=True)
        
        # 加载模型和数据
        model = load_model()
        df = load_data()
        
        # 分离特征和目标变量
        X = df.drop(columns=['travel_time'])
        y = df['travel_time']
        
        # 基本评估指标
        metrics, y_pred = evaluate_basic_metrics(model, X, y)
        
        # 交叉验证评估
        cv_results = evaluate_cross_validation(model, X, y, cv=5)
        
        # 残差分析
        residuals, residual_stats = analyze_residuals(y, y_pred)
        
        # 创建评估可视化
        visualization_paths = create_evaluation_visualizations(y, y_pred, residuals)
        
        # 分析特征影响
        shap_results = analyze_feature_impact(model, X)
        
        # 分析错误分布
        error_results = analyze_error_distribution(y, y_pred, X)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存评估结果
        evaluation_results = {
            'basic_metrics': metrics,
            'cv_results': cv_results,
            'residual_stats': residual_stats,
            'visualization_paths': visualization_paths,
            'worst_samples': error_results['worst_samples'].to_dict(),
            'error_correlations': error_results['error_correlations'].to_dict()
        }
        
        # 保存评估摘要
        evaluation_summary_path = EVALUATION_DIR / f"evaluation_summary_{timestamp}.txt"
        with open(evaluation_summary_path, 'w') as f:
            f.write("模型评估摘要\n")
            f.write("===========\n\n")
            
            f.write("基本评估指标:\n")
            f.write(f"平均绝对误差 (MAE): {metrics['mae']:.4f}\n")
            f.write(f"均方误差 (MSE): {metrics['mse']:.4f}\n")
            f.write(f"均方根误差 (RMSE): {metrics['rmse']:.4f}\n")
            f.write(f"决定系数 (R²): {metrics['r2']:.4f}\n\n")
            
            f.write("交叉验证结果:\n")
            for score, result in cv_results.items():
                f.write(f"{score}: {result['mean']:.4f} ± {result['std']:.4f}\n")
            f.write("\n")
            
            f.write("残差统计信息:\n")
            for stat, value in residual_stats.items():
                f.write(f"{stat}: {value:.4f}\n")
            f.write("\n")
            
            f.write("生成的可视化:\n")
            for name, path in visualization_paths.items():
                f.write(f"{name}: {path}\n")
            f.write("\n")
            
            f.write("与误差最相关的特征 (Top 5):\n")
            for i, row in error_results['error_correlations'].head(5).iterrows():
                f.write(f"{row['feature']}: {row['correlation_with_error']:.4f}\n")
        
        logger.info(f"评估摘要已保存至: {evaluation_summary_path}")
        logger.info("模型评估流程完成")
        
    except Exception as e:
        logger.error(f"模型评估过程中发生错误: {e}")
        raise


if __name__ == '__main__':
    main() 