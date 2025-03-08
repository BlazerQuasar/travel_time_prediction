#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主脚本：运行完整的旅行时间预测工作流程。
按顺序执行数据处理、特征工程、模型训练和评估等步骤。
"""

import os
import logging
import argparse
from pathlib import Path
import sys

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent


def setup_directories():
    """创建必要的项目目录"""
    logger.info("创建项目目录...")
    
    # 定义目录路径
    data_dir = ROOT_DIR / 'data'
    raw_data_dir = data_dir / 'raw'
    processed_data_dir = data_dir / 'processed'
    features_dir = processed_data_dir / 'features'
    models_dir = ROOT_DIR / 'models'
    evaluation_dir = ROOT_DIR / 'evaluation'
    predictions_dir = ROOT_DIR / 'predictions'
    
    # 创建目录
    dirs_to_create = [
        raw_data_dir, 
        processed_data_dir, 
        features_dir, 
        models_dir, 
        evaluation_dir, 
        predictions_dir
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"已创建目录: {dir_path}")
    
    return {
        'raw_data': raw_data_dir,
        'processed_data': processed_data_dir,
        'features': features_dir,
        'models': models_dir,
        'evaluation': evaluation_dir,
        'predictions': predictions_dir
    }


def generate_example_data(dirs):
    """生成示例数据（如果不存在）"""
    logger.info("检查并生成示例数据...")
    
    # 导入数据生成模块
    from src.data.make_dataset import create_example_traffic_data, create_example_weather_data, create_example_holiday_data
    
    # 定义文件路径
    traffic_filepath = dirs['raw_data'] / 'traffic_data.csv'
    weather_filepath = dirs['raw_data'] / 'weather_data.csv'
    holiday_filepath = dirs['raw_data'] / 'holiday_data.csv'
    
    # 生成数据（如果不存在）
    if not traffic_filepath.exists():
        logger.info("生成示例交通数据...")
        create_example_traffic_data(traffic_filepath)
    
    if not weather_filepath.exists():
        logger.info("生成示例天气数据...")
        create_example_weather_data(weather_filepath)
    
    if not holiday_filepath.exists():
        logger.info("生成示例假日数据...")
        create_example_holiday_data(holiday_filepath)
    
    logger.info("示例数据检查/生成完成")


def run_data_processing():
    """运行数据处理步骤"""
    logger.info("运行数据处理...")
    
    # 导入数据处理模块
    from src.data.make_dataset import main as process_data
    
    # 执行数据处理
    process_data()
    
    logger.info("数据处理完成")


def run_feature_engineering():
    """运行特征工程步骤"""
    logger.info("运行特征工程...")
    
    # 导入特征工程模块
    from src.features.build_features import main as build_features
    
    # 执行特征工程
    build_features()
    
    logger.info("特征工程完成")


def run_model_training(optimize_hyperparams=True):
    """运行模型训练步骤"""
    logger.info("运行模型训练...")
    
    # 修改src/models/train_model.py中的超参数优化标志
    from src.models import train_model
    
    # 保存原始值
    original_value = train_model.do_hyperparameter_optimization if hasattr(train_model, 'do_hyperparameter_optimization') else True
    
    # 设置超参数优化标志
    train_model.do_hyperparameter_optimization = optimize_hyperparams
    
    # 执行模型训练
    train_model.main()
    
    # 恢复原始值
    if hasattr(train_model, 'do_hyperparameter_optimization'):
        train_model.do_hyperparameter_optimization = original_value
    
    logger.info("模型训练完成")


def run_model_evaluation():
    """运行模型评估步骤"""
    logger.info("运行模型评估...")
    
    # 导入模型评估模块
    from src.models.evaluate_model import main as evaluate_model
    
    # 执行模型评估
    evaluate_model()
    
    logger.info("模型评估完成")


def run_prediction_example():
    """运行预测示例"""
    logger.info("运行预测示例...")
    
    # 导入预测模块
    from src.models.predict_model import predict_from_features
    
    # 使用特征数据进行预测
    processed_data_path = ROOT_DIR / 'data' / 'processed' / 'features' / 'final_features.csv'
    
    if processed_data_path.exists():
        predict_from_features(processed_data_path)
        logger.info("预测示例完成")
    else:
        logger.warning(f"无法运行预测示例，特征数据文件不存在: {processed_data_path}")


def run_full_pipeline(optimize_hyperparams=False):
    """运行完整的工作流程"""
    logger.info("启动完整的旅行时间预测工作流程")
    
    try:
        # 设置目录
        dirs = setup_directories()
        
        # 生成示例数据
        generate_example_data(dirs)
        
        # 数据处理
        run_data_processing()
        
        # 特征工程
        run_feature_engineering()
        
        # 模型训练
        run_model_training(optimize_hyperparams)
        
        # 模型评估
        run_model_evaluation()
        
        # 预测示例
        run_prediction_example()
        
        logger.info("工作流程执行完成!")
        
    except Exception as e:
        logger.error(f"工作流程执行失败: {e}")
        raise


def main():
    """主函数：解析命令行参数并执行相应的步骤"""
    parser = argparse.ArgumentParser(description='旅行时间预测工作流程执行器')
    
    parser.add_argument('--full', action='store_true', 
                        help='运行完整的工作流程')
    parser.add_argument('--optimize', action='store_true', 
                        help='执行超参数优化（与--full一起使用，默认为False）')
    parser.add_argument('--data', action='store_true', 
                        help='仅运行数据处理步骤')
    parser.add_argument('--features', action='store_true', 
                        help='仅运行特征工程步骤')
    parser.add_argument('--train', action='store_true', 
                        help='仅运行模型训练步骤')
    parser.add_argument('--evaluate', action='store_true', 
                        help='仅运行模型评估步骤')
    parser.add_argument('--predict', action='store_true', 
                        help='仅运行预测示例')
    
    args = parser.parse_args()
    
    # 添加src目录到Python路径
    sys.path.append(str(ROOT_DIR))
    
    # 根据参数执行相应的步骤
    if args.full or not any([args.data, args.features, args.train, args.evaluate, args.predict]):
        run_full_pipeline(args.optimize)
    else:
        if args.data:
            setup_directories()
            generate_example_data(setup_directories())
            run_data_processing()
        
        if args.features:
            run_feature_engineering()
        
        if args.train:
            run_model_training(args.optimize)
        
        if args.evaluate:
            run_model_evaluation()
        
        if args.predict:
            run_prediction_example()
    
    return 0


if __name__ == '__main__':
    exit(main()) 