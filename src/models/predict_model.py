#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
预测脚本：使用训练好的模型进行旅行时间预测。
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / 'models'
DATA_DIR = ROOT_DIR / 'data' / 'processed' / 'features'
PREDICTIONS_DIR = ROOT_DIR / 'predictions'


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


def load_prediction_data(filepath):
    """加载需要预测的数据"""
    logger.info(f"加载预测数据: {filepath}")
    
    if not Path(filepath).exists():
        logger.error(f"预测数据文件不存在: {filepath}")
        raise FileNotFoundError(f"找不到预测数据文件: {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"成功加载预测数据，包含 {len(df)} 条记录")
    return df


def preprocess_prediction_data(df, encoders_path=None):
    """预处理预测数据"""
    logger.info("预处理预测数据...")
    
    # 如果有需要使用的编码器，加载它们
    encoders = None
    if encoders_path:
        if Path(encoders_path).exists():
            logger.info(f"加载编码器: {encoders_path}")
            encoders = joblib.load(encoders_path)
        else:
            logger.warning(f"编码器文件不存在: {encoders_path}")
    
    # 假设这里进行与训练数据相同的预处理步骤
    # 例如特征工程、数据转换等
    # 这个函数的具体实现取决于模型训练时的预处理步骤
    
    # 示例：应用独热编码
    if encoders:
        for feature, encoder in encoders.items():
            if feature in df.columns:
                # 转换数据
                encoded = encoder.transform(df[[feature]])
                
                # 创建列名
                feature_names = [f"{feature}_{val}" for val in encoder.categories_[0]]
                
                # 添加编码后的列
                encoded_df = pd.DataFrame(encoded, columns=feature_names)
                
                # 将编码后的列添加到原始DataFrame
                df = pd.concat([df.drop(columns=[feature]), encoded_df], axis=1)
    
    logger.info("预处理完成")
    return df


def make_predictions(model, X):
    """使用模型进行预测"""
    logger.info("开始进行预测...")
    
    # 确保数据中只包含模型所需的特征
    model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
    
    if model_features is not None:
        missing_features = set(model_features) - set(X.columns)
        extra_features = set(X.columns) - set(model_features)
        
        if missing_features:
            logger.warning(f"数据中缺少模型所需的特征: {missing_features}")
            raise ValueError(f"数据中缺少模型所需的特征: {missing_features}")
        
        if extra_features:
            logger.warning(f"数据中包含模型不需要的特征，这些特征将被忽略: {extra_features}")
            X = X[model_features]
    
    # 进行预测
    predictions = model.predict(X)
    
    logger.info(f"预测完成，共 {len(predictions)} 条预测结果")
    return predictions


def save_predictions(df, predictions, output_filepath=None):
    """保存预测结果"""
    logger.info("保存预测结果...")
    
    # 创建包含预测结果的DataFrame
    result_df = df.copy()
    result_df['predicted_travel_time'] = predictions
    
    # 如果没有指定输出文件路径，则生成一个默认路径
    if output_filepath is None:
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filepath = PREDICTIONS_DIR / f"predictions_{timestamp}.csv"
    
    # 保存预测结果
    result_df.to_csv(output_filepath, index=False)
    
    logger.info(f"预测结果已保存至: {output_filepath}")
    return output_filepath


def predict_from_features(input_filepath, output_filepath=None):
    """从特征数据进行预测的主要功能"""
    logger.info(f"从特征数据进行预测: {input_filepath}")
    
    try:
        # 加载模型
        model = load_model()
        
        # 加载预测数据
        df = load_prediction_data(input_filepath)
        
        # 加载编码器（如果存在）
        encoders_path = DATA_DIR / 'categorical_encoders.joblib'
        if encoders_path.exists():
            df = preprocess_prediction_data(df, encoders_path)
        else:
            df = preprocess_prediction_data(df)
        
        # 检查并清理数据（例如，删除目标列如果存在）
        if 'travel_time' in df.columns:
            logger.info("预测数据中包含目标列 'travel_time'，将被忽略")
            target_values = df['travel_time'].copy()  # 保存实际值用于可能的比较
            X = df.drop(columns=['travel_time'])
        else:
            X = df.copy()
            target_values = None
        
        # 进行预测
        predictions = make_predictions(model, X)
        
        # 保存预测结果
        saved_path = save_predictions(df, predictions, output_filepath)
        
        # 如果有实际值，可以计算预测精度
        if target_values is not None:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            mae = mean_absolute_error(target_values, predictions)
            rmse = np.sqrt(mean_squared_error(target_values, predictions))
            r2 = r2_score(target_values, predictions)
            
            logger.info(f"预测精度评估:")
            logger.info(f"平均绝对误差 (MAE): {mae:.4f}")
            logger.info(f"均方根误差 (RMSE): {rmse:.4f}")
            logger.info(f"决定系数 (R²): {r2:.4f}")
        
        logger.info(f"预测成功完成，结果已保存至: {saved_path}")
        return saved_path
        
    except Exception as e:
        logger.error(f"预测过程中发生错误: {e}")
        raise


def predict_from_raw_data(input_filepath, output_filepath=None):
    """从原始数据进行预测，需要进行完整的数据处理和特征工程"""
    logger.info(f"从原始数据进行预测: {input_filepath}")
    
    try:
        # 这里需要复用之前创建的数据处理和特征工程流程
        # 由于这需要更复杂的实现，这里仅提供一个框架
        
        # 导入必要的处理模块
        import sys
        sys.path.append(str(ROOT_DIR / 'src'))
        
        from data.make_dataset import clean_traffic_data
        from features.build_features import (
            create_time_features, create_weather_features, 
            create_traffic_features, create_combined_features,
            handle_categorical_features
        )
        
        # 加载原始数据
        df = pd.read_csv(input_filepath)
        
        # 清洗数据
        df = clean_traffic_data(df)
        
        # 特征工程
        df = create_time_features(df)
        df = create_weather_features(df)
        df = create_traffic_features(df)
        df = create_combined_features(df)
        
        # 处理分类特征
        df = handle_categorical_features(df)
        
        # 加载特征选择结果，确保只使用重要特征
        feature_selection_path = DATA_DIR / 'feature_selection_results.joblib'
        if feature_selection_path.exists():
            feature_selection = joblib.load(feature_selection_path)
            selected_features = feature_selection['final_features']
            
            # 确保所有选定的特征都存在
            missing_features = set(selected_features) - set(df.columns)
            if missing_features:
                logger.warning(f"数据中缺少一些选定的特征: {missing_features}")
                # 这里可以采取一些措施，如用默认值填充
            
            # 只保留选定的特征
            df = df[[col for col in selected_features if col in df.columns]]
        
        # 调用前面定义的函数进行预测
        return predict_from_features(df, output_filepath)
        
    except Exception as e:
        logger.error(f"从原始数据预测过程中发生错误: {e}")
        raise


def main():
    """主函数：根据命令行参数执行预测"""
    import argparse
    
    parser = argparse.ArgumentParser(description='使用训练好的模型进行旅行时间预测')
    parser.add_argument('input_file', help='输入数据文件路径')
    parser.add_argument('--output', '-o', help='输出预测结果文件路径')
    parser.add_argument('--raw', '-r', action='store_true', help='输入是原始数据（需要完整的处理）')
    
    args = parser.parse_args()
    
    try:
        if args.raw:
            predict_from_raw_data(args.input_file, args.output)
        else:
            predict_from_features(args.input_file, args.output)
    except Exception as e:
        logger.error(f"预测失败: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main()) 