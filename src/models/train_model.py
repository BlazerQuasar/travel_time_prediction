#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型训练脚本：训练随机森林回归器和超参数优化。
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / 'data' / 'processed' / 'features'
MODELS_DIR = ROOT_DIR / 'models'


def load_feature_data():
    """加载特征工程后的数据"""
    logger.info("加载特征数据...")
    
    filepath = DATA_DIR / 'final_features.csv'
    if not filepath.exists():
        logger.error(f"特征数据文件不存在: {filepath}")
        raise FileNotFoundError(f"找不到特征数据文件: {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"成功加载特征数据，包含 {len(df)} 条记录和 {df.shape[1]} 个特征")
    return df


def split_data(df, target_column='travel_time', test_size=0.2, random_state=42):
    """划分训练集和测试集"""
    logger.info(f"划分数据集，测试集比例: {test_size}")
    
    # 分离特征和目标变量
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_base_model(X_train, y_train, random_state=42):
    """训练基础随机森林模型"""
    logger.info("训练基础随机森林回归模型...")
    
    # 创建模型
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=random_state,
        n_jobs=-1  # 使用所有可用的CPU核心
    )
    
    # 训练模型
    model.fit(X_train, y_train)
    
    logger.info("基础模型训练完成")
    return model


def optimize_hyperparameters(X_train, y_train, random_state=42, cv=5):
    """超参数优化"""
    logger.info("开始超参数优化...")
    
    # 定义参数网格
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # 创建基础模型
    base_model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    
    # 创建网格搜索
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    # 执行网格搜索
    logger.info("执行网格搜索，这可能需要一些时间...")
    grid_search.fit(X_train, y_train)
    
    # 获取最佳参数和模型
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    logger.info(f"最佳参数: {best_params}")
    logger.info(f"最佳交叉验证分数: {-grid_search.best_score_}")
    
    return best_model, best_params, grid_search


def train_final_model(X_train, y_train, best_params, random_state=42):
    """使用最佳参数训练最终模型"""
    logger.info("使用最佳参数训练最终模型...")
    
    # 创建模型
    model = RandomForestRegressor(
        **best_params,
        random_state=random_state,
        n_jobs=-1
    )
    
    # 训练模型
    model.fit(X_train, y_train)
    
    logger.info("最终模型训练完成")
    return model


def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    logger.info("评估模型性能...")
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # 输出评估结果
    logger.info(f"平均绝对误差 (MAE): {mae:.4f}")
    logger.info(f"均方根误差 (RMSE): {rmse:.4f}")
    logger.info(f"决定系数 (R²): {r2:.4f}")
    
    # 返回评估指标
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def analyze_feature_importance(model, X_train):
    """分析特征重要性"""
    logger.info("分析特征重要性...")
    
    # 获取特征重要性
    feature_importances = model.feature_importances_
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': feature_importances
    })
    
    # 按重要性排序
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # 输出前10个最重要的特征
    logger.info("前10个最重要的特征:")
    for i, row in importance_df.head(10).iterrows():
        logger.info(f"{row['feature']}: {row['importance']:.4f}")
    
    return importance_df


def save_model(model, metrics, feature_importance_df):
    """保存模型和相关信息"""
    logger.info("保存模型和相关信息...")
    
    # 创建模型目录
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存模型
    model_path = MODELS_DIR / f"random_forest_model_{timestamp}.joblib"
    joblib.dump(model, model_path)
    
    # 保存最新模型的链接
    latest_model_path = MODELS_DIR / "latest_model.joblib"
    if os.path.exists(latest_model_path):
        os.remove(latest_model_path)
    joblib.dump(model, latest_model_path)
    
    # 保存评估指标
    metrics_path = MODELS_DIR / f"model_metrics_{timestamp}.json"
    pd.Series(metrics).to_json(metrics_path)
    
    # 保存特征重要性
    importance_path = MODELS_DIR / f"feature_importance_{timestamp}.csv"
    feature_importance_df.to_csv(importance_path, index=False)
    
    logger.info(f"模型已保存至: {model_path}")
    logger.info(f"评估指标已保存至: {metrics_path}")
    logger.info(f"特征重要性已保存至: {importance_path}")


def main():
    """主函数：执行模型训练流程"""
    logger.info("开始模型训练流程")
    
    try:
        # 加载特征数据
        df = load_feature_data()
        
        # 划分数据集
        X_train, X_test, y_train, y_test = split_data(df, target_column='travel_time')
        
        # 是否执行超参数优化
        do_hyperparameter_optimization = True
        
        if do_hyperparameter_optimization:
            # 超参数优化（警告：这可能需要较长时间）
            best_model, best_params, grid_search = optimize_hyperparameters(X_train, y_train)
            
            # 使用最佳参数训练最终模型
            model = train_final_model(X_train, y_train, best_params)
        else:
            # 训练基础模型
            model = train_base_model(X_train, y_train)
        
        # 评估模型
        metrics = evaluate_model(model, X_test, y_test)
        
        # 分析特征重要性
        feature_importance_df = analyze_feature_importance(model, X_train)
        
        # 保存模型和相关信息
        save_model(model, metrics, feature_importance_df)
        
        logger.info("模型训练流程完成")
        
    except Exception as e:
        logger.error(f"模型训练过程中发生错误: {e}")
        raise


if __name__ == '__main__':
    main() 