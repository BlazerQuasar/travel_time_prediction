#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
特征工程脚本：创建和选择模型所需的特征。
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'
OUTPUT_DIR = PROCESSED_DATA_DIR / 'features'


def load_processed_data():
    """加载预处理后的数据"""
    logger.info("加载预处理后的数据...")
    
    filepath = PROCESSED_DATA_DIR / 'processed_data.csv'
    if not filepath.exists():
        logger.error(f"预处理数据文件不存在: {filepath}")
        raise FileNotFoundError(f"找不到预处理数据文件: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # 确保时间戳列为日期时间类型
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    logger.info(f"成功加载预处理数据，包含 {len(df)} 条记录和 {df.shape[1]} 个特征")
    return df


def create_time_features(df):
    """创建与时间相关的特征"""
    logger.info("创建时间特征...")
    
    # 确保有timestamp列
    if 'timestamp' not in df.columns:
        logger.warning("数据中没有timestamp列，无法创建时间特征")
        return df
    
    # 提取时间特征
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=周一，6=周日
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # 是否为周末
    
    # 定义高峰时段
    # 早高峰: 7:00-9:00, 晚高峰: 17:00-19:00
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    df['is_rush_hour'] = ((df['is_morning_rush'] == 1) | (df['is_evening_rush'] == 1)).astype(int)
    
    # 定义时段
    # 0:00-6:00 = 深夜, 6:00-12:00 = 上午, 12:00-18:00 = 下午, 18:00-24:00 = 晚上
    conditions = [
        (df['hour'] >= 0) & (df['hour'] < 6),
        (df['hour'] >= 6) & (df['hour'] < 12),
        (df['hour'] >= 12) & (df['hour'] < 18),
        (df['hour'] >= 18) & (df['hour'] <= 23)
    ]
    choices = [0, 1, 2, 3]  # 对应深夜、上午、下午、晚上
    df['time_of_day'] = np.select(conditions, choices, default=np.nan)
    
    logger.info("时间特征创建完成")
    return df


def create_weather_features(df):
    """创建与天气相关的特征"""
    logger.info("创建天气特征...")
    
    if 'weather_condition' not in df.columns or 'precipitation' not in df.columns:
        logger.warning("数据中缺少天气相关列，无法创建天气特征")
        return df
    
    # 是否有降水（雨或雪）
    if 'precipitation' in df.columns:
        df['has_precipitation'] = (df['precipitation'] > 0).astype(int)
    
    # 气温分类
    if 'temperature' in df.columns:
        conditions = [
            (df['temperature'] < 0),            # 冰冻
            (df['temperature'] >= 0) & (df['temperature'] < 10),  # 寒冷
            (df['temperature'] >= 10) & (df['temperature'] < 20), # 凉爽
            (df['temperature'] >= 20) & (df['temperature'] < 30), # 温暖
            (df['temperature'] >= 30)           # 炎热
        ]
        choices = [0, 1, 2, 3, 4]
        df['temperature_category'] = np.select(conditions, choices, default=np.nan)
    
    # 天气状况编码（如果是分类变量）
    if 'weather_condition' in df.columns:
        # 假设weather_condition是编码过的：0=晴天，1=多云，2=雨天，3=雪天
        # 如果不是，这里需要进行适当的编码或变换
        # 如果是文本形式，可以使用OneHotEncoder进行编码
        pass
    
    logger.info("天气特征创建完成")
    return df


def create_traffic_features(df):
    """创建与交通相关的特征"""
    logger.info("创建交通特征...")
    
    # 创建拥堵指标
    if 'avg_speed' in df.columns and 'traffic_flow' in df.columns:
        # 交通密度 = 流量/速度
        df['traffic_density'] = df['traffic_flow'] / df['avg_speed'].replace(0, np.nan)
        
        # 分类交通拥堵等级
        if 'avg_speed' in df.columns:
            conditions = [
                (df['avg_speed'] < 20),                          # 严重拥堵
                (df['avg_speed'] >= 20) & (df['avg_speed'] < 40),  # 中度拥堵
                (df['avg_speed'] >= 40) & (df['avg_speed'] < 60),  # 轻微拥堵
                (df['avg_speed'] >= 60)                          # 畅通
            ]
            choices = [0, 1, 2, 3]
            df['congestion_level'] = np.select(conditions, choices, default=np.nan)
    
    # 计算每个道路段的历史平均速度和旅行时间
    if 'road_segment_id' in df.columns and 'avg_speed' in df.columns and 'travel_time' in df.columns:
        road_stats = df.groupby('road_segment_id').agg({
            'avg_speed': 'mean',
            'travel_time': 'mean'
        }).reset_index()
        
        road_stats.columns = ['road_segment_id', 'road_avg_speed', 'road_avg_travel_time']
        
        # 合并回原始数据
        df = pd.merge(df, road_stats, on='road_segment_id', how='left')
        
        # 计算当前速度与道路平均速度的比率
        df['speed_ratio'] = df['avg_speed'] / df['road_avg_speed']
    
    logger.info("交通特征创建完成")
    return df


def create_combined_features(df):
    """创建组合特征"""
    logger.info("创建组合特征...")
    
    # 道路和时间段组合
    if 'road_segment_id' in df.columns and 'time_of_day' in df.columns:
        df['road_time_group'] = df['road_segment_id'].astype(str) + '_' + df['time_of_day'].astype(str)
    
    # 天气和时间段组合
    if 'weather_condition' in df.columns and 'is_rush_hour' in df.columns:
        df['weather_rush_group'] = df['weather_condition'].astype(str) + '_' + df['is_rush_hour'].astype(str)
    
    # 工作日/周末与高峰时段组合
    if 'is_weekend' in df.columns and 'is_rush_hour' in df.columns:
        df['weekend_rush_group'] = df['is_weekend'].astype(str) + '_' + df['is_rush_hour'].astype(str)
    
    logger.info("组合特征创建完成")
    return df


def handle_categorical_features(df):
    """处理分类特征（编码等）"""
    logger.info("处理分类特征...")
    
    # 定义需要进行独热编码的分类特征
    categorical_features = [col for col in df.columns if 
                           df[col].dtype == 'object' or 
                           col in ['road_segment_id', 'weather_condition', 'time_of_day', 'road_time_group', 
                                 'weather_rush_group', 'weekend_rush_group']]
    
    encoded_dfs = []
    encoders = {}
    
    # 保留原始DataFrame中不需要编码的列
    non_categorical = [col for col in df.columns if col not in categorical_features]
    encoded_dfs.append(df[non_categorical].reset_index(drop=True))
    
    # 对每个分类特征进行独热编码
    for feature in categorical_features:
        if feature in df.columns:
            # 创建编码器
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            
            # 转换数据
            encoded = encoder.fit_transform(df[[feature]])
            
            # 创建列名
            feature_names = [f"{feature}_{val}" for val in encoder.categories_[0]]
            
            # 创建DataFrame
            encoded_df = pd.DataFrame(encoded, columns=feature_names)
            
            # 添加到结果列表
            encoded_dfs.append(encoded_df.reset_index(drop=True))
            
            # 保存编码器
            encoders[feature] = encoder
    
    # 合并所有编码后的DataFrame
    result_df = pd.concat(encoded_dfs, axis=1)
    
    # 保存编码器
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(encoders, OUTPUT_DIR / 'categorical_encoders.joblib')
    
    logger.info(f"分类特征处理完成，处理后特征数量: {result_df.shape[1]}")
    return result_df


def select_features(df, target_column='travel_time', n_features=20):
    """特征选择"""
    logger.info("开始进行特征选择...")
    
    # 确保目标列存在
    if target_column not in df.columns:
        logger.error(f"目标列 {target_column} 不在数据集中")
        raise ValueError(f"目标列 {target_column} 不在数据集中")
    
    # 分离特征和目标变量
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 删除不应作为特征的列
    cols_to_drop = ['timestamp', 'date'] if 'timestamp' in X.columns else []
    X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])
    
    # 计算互信息
    logger.info("计算特征的互信息分数...")
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_df = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores})
    mi_df = mi_df.sort_values('mi_score', ascending=False)
    
    # 使用随机森林的特征重要性
    logger.info("使用随机森林计算特征重要性...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importances = rf.feature_importances_
    rf_df = pd.DataFrame({'feature': X.columns, 'rf_importance': rf_importances})
    rf_df = rf_df.sort_values('rf_importance', ascending=False)
    
    # 使用递归特征消除
    logger.info("使用递归特征消除...")
    rfe = RFE(estimator=RandomForestRegressor(n_estimators=50, random_state=42), 
              n_features_to_select=n_features)
    rfe.fit(X, y)
    rfe_selected = X.columns[rfe.support_]
    
    # 汇总特征选择结果
    logger.info("汇总特征选择结果...")
    feature_selection_results = {
        'mutual_info_top_features': mi_df.head(n_features)['feature'].tolist(),
        'random_forest_top_features': rf_df.head(n_features)['feature'].tolist(),
        'rfe_selected_features': rfe_selected.tolist()
    }
    
    # 选择最终的特征集
    # 将三种方法选出的特征进行合并，优先考虑在多个方法中都被选中的特征
    all_selected = set()
    for method, features in feature_selection_results.items():
        all_selected.update(features)
    
    # 为每个特征计算被选择的次数
    feature_counts = {}
    for feature in all_selected:
        count = sum(1 for method_features in feature_selection_results.values() if feature in method_features)
        feature_counts[feature] = count
    
    # 按被选择次数排序，选择top n_features
    final_selected = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:n_features]
    final_features = [feature for feature, count in final_selected]
    
    # 保存特征选择结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    feature_selection_path = OUTPUT_DIR / 'feature_selection_results.joblib'
    joblib.dump({
        'all_results': feature_selection_results,
        'final_features': final_features
    }, feature_selection_path)
    
    logger.info(f"特征选择完成，选择了 {len(final_features)} 个特征")
    return final_features


def prepare_final_dataset(df, selected_features, target_column='travel_time'):
    """准备最终数据集"""
    logger.info("准备最终数据集...")
    
    # 提取选定的特征和目标变量
    final_cols = selected_features + [target_column]
    final_df = df[final_cols].copy()
    
    # 保存最终数据集
    final_data_path = OUTPUT_DIR / 'final_features.csv'
    final_df.to_csv(final_data_path, index=False)
    
    logger.info(f"最终数据集已保存至: {final_data_path}")
    return final_df


def main():
    """主函数：执行特征工程流程"""
    logger.info("开始特征工程流程")
    
    try:
        # 创建输出目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 加载预处理后的数据
        df = load_processed_data()
        
        # 特征创建
        df = create_time_features(df)
        df = create_weather_features(df)
        df = create_traffic_features(df)
        df = create_combined_features(df)
        
        # 处理分类特征
        df_encoded = handle_categorical_features(df)
        
        # 特征选择
        selected_features = select_features(df_encoded, target_column='travel_time', n_features=20)
        
        # 准备最终数据集
        final_df = prepare_final_dataset(df_encoded, selected_features, target_column='travel_time')
        
        logger.info("特征工程流程完成")
        
    except Exception as e:
        logger.error(f"特征工程过程中发生错误: {e}")
        raise


if __name__ == '__main__':
    import os  # 导入os模块，用于创建目录
    main() 