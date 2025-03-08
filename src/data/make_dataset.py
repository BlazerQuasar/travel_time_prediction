#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据处理脚本：加载、清洗和预处理原始数据。
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'


def load_traffic_data(filepath):
    """加载交通数据"""
    logger.info(f"加载交通数据: {filepath}")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"成功加载 {len(df)} 条交通记录")
        return df
    except Exception as e:
        logger.error(f"加载交通数据失败: {e}")
        raise


def load_weather_data(filepath):
    """加载天气数据"""
    logger.info(f"加载天气数据: {filepath}")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"成功加载 {len(df)} 条天气记录")
        return df
    except Exception as e:
        logger.error(f"加载天气数据失败: {e}")
        raise


def load_holiday_data(filepath):
    """加载假日数据"""
    logger.info(f"加载假日数据: {filepath}")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"成功加载 {len(df)} 条假日记录")
        return df
    except Exception as e:
        logger.error(f"加载假日数据失败: {e}")
        raise


def clean_traffic_data(df):
    """清洗交通数据"""
    logger.info("清洗交通数据...")
    
    # 删除重复项
    df = df.drop_duplicates()
    
    # 转换时间戳
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 处理缺失值
    # 这里根据具体数据情况选择合适的策略，如填充、删除等
    
    # 检测并处理异常值
    # 例如，使用IQR方法检测异常值
    for col in df.select_dtypes(include=[np.number]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # 将异常值替换为边界值
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    logger.info(f"交通数据清洗完成，剩余 {len(df)} 条记录")
    return df


def clean_weather_data(df):
    """清洗天气数据"""
    logger.info("清洗天气数据...")
    
    # 删除重复项
    df = df.drop_duplicates()
    
    # 转换时间戳
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 处理缺失值 - 根据具体情况选择策略
    
    logger.info(f"天气数据清洗完成，剩余 {len(df)} 条记录")
    return df


def merge_datasets(traffic_df, weather_df, holiday_df):
    """合并数据集"""
    logger.info("合并数据集...")
    
    # 假设三个数据集都有timestamp列作为合并的键
    # 用左连接保留所有交通记录
    merged_df = pd.merge(
        traffic_df,
        weather_df,
        on='timestamp',
        how='left',
        suffixes=('', '_weather')
    )
    
    # 合并假日数据
    # 假设holiday_df含有日期列'date'
    if 'date' in holiday_df.columns:
        merged_df['date'] = merged_df['timestamp'].dt.date
        merged_df = pd.merge(
            merged_df,
            holiday_df,
            on='date',
            how='left',
            suffixes=('', '_holiday')
        )
    
    logger.info(f"数据集合并完成，最终数据集包含 {len(merged_df)} 条记录和 {merged_df.shape[1]} 个特征")
    return merged_df


def main():
    """主函数：执行数据处理流程"""
    logger.info("开始数据处理流程")
    
    # 确保输出目录存在
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # 假设路径，实际使用时请替换为真实文件路径
    traffic_filepath = RAW_DATA_DIR / 'traffic_data.csv'
    weather_filepath = RAW_DATA_DIR / 'weather_data.csv'
    holiday_filepath = RAW_DATA_DIR / 'holiday_data.csv'
    
    try:
        # 检查文件是否存在
        if not traffic_filepath.exists():
            logger.warning(f"交通数据文件不存在: {traffic_filepath}")
            # 在这里可以添加示例数据的生成
            create_example_traffic_data(traffic_filepath)
        
        if not weather_filepath.exists():
            logger.warning(f"天气数据文件不存在: {weather_filepath}")
            create_example_weather_data(weather_filepath)
            
        if not holiday_filepath.exists():
            logger.warning(f"假日数据文件不存在: {holiday_filepath}")
            create_example_holiday_data(holiday_filepath)
        
        # 加载数据
        traffic_df = load_traffic_data(traffic_filepath)
        weather_df = load_weather_data(weather_filepath)
        holiday_df = load_holiday_data(holiday_filepath)
        
        # 清洗数据
        traffic_df = clean_traffic_data(traffic_df)
        weather_df = clean_weather_data(weather_df)
        
        # 合并数据集
        final_df = merge_datasets(traffic_df, weather_df, holiday_df)
        
        # 保存处理后的数据
        output_filepath = PROCESSED_DATA_DIR / 'processed_data.csv'
        final_df.to_csv(output_filepath, index=False)
        logger.info(f"处理后的数据已保存至: {output_filepath}")
        
    except Exception as e:
        logger.error(f"数据处理过程中发生错误: {e}")
        raise
    
    logger.info("数据处理流程完成")


def create_example_traffic_data(filepath):
    """创建示例交通数据（仅用于演示）"""
    logger.info(f"创建示例交通数据: {filepath}")
    
    # 确保目录存在
    os.makedirs(filepath.parent, exist_ok=True)
    
    # 创建日期范围
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 1, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # 创建随机数据
    np.random.seed(42)  # 设置随机种子以确保可重复性
    n_samples = len(dates)
    
    data = {
        'timestamp': dates,
        'road_segment_id': np.random.randint(1, 11, n_samples),
        'travel_time': np.random.normal(15, 5, n_samples),  # 平均15分钟，标准差5分钟
        'traffic_flow': np.random.poisson(100, n_samples),  # 平均每小时100辆车
        'avg_speed': np.random.normal(60, 15, n_samples),   # 平均时速60公里，标准差15公里
    }
    
    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    logger.info(f"已创建示例交通数据，包含 {len(df)} 条记录")


def create_example_weather_data(filepath):
    """创建示例天气数据（仅用于演示）"""
    logger.info(f"创建示例天气数据: {filepath}")
    
    # 确保目录存在
    os.makedirs(filepath.parent, exist_ok=True)
    
    # 创建日期范围（每3小时一条记录）
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 1, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='3H')
    
    # 创建随机数据
    np.random.seed(43)  # 不同的随机种子
    n_samples = len(dates)
    
    # 天气状况：0=晴天，1=多云，2=雨天，3=雪天
    weather_conditions = np.random.choice([0, 1, 2, 3], n_samples, p=[0.5, 0.3, 0.15, 0.05])
    
    data = {
        'timestamp': dates,
        'temperature': np.random.normal(15, 10, n_samples),  # 摄氏度
        'precipitation': np.random.exponential(0.5, n_samples) * (weather_conditions >= 2),  # 降水量（毫米）
        'wind_speed': np.random.exponential(5, n_samples),   # 风速（公里/小时）
        'weather_condition': weather_conditions
    }
    
    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    logger.info(f"已创建示例天气数据，包含 {len(df)} 条记录")


def create_example_holiday_data(filepath):
    """创建示例假日数据（仅用于演示）"""
    logger.info(f"创建示例假日数据: {filepath}")
    
    # 确保目录存在
    os.makedirs(filepath.parent, exist_ok=True)
    
    # 2022年的一些假日（示例）
    holidays = [
        {'date': '2022-01-01', 'holiday_name': '元旦', 'is_holiday': 1},
        {'date': '2022-01-02', 'holiday_name': '元旦假期', 'is_holiday': 1},
        {'date': '2022-01-03', 'holiday_name': '元旦假期', 'is_holiday': 1},
        {'date': '2022-01-31', 'holiday_name': '春节', 'is_holiday': 1},
    ]
    
    # 创建DataFrame并保存
    df = pd.DataFrame(holidays)
    df.to_csv(filepath, index=False)
    logger.info(f"已创建示例假日数据，包含 {len(df)} 条记录")


if __name__ == '__main__':
    main() 