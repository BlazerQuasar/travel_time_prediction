#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
环境设置脚本：创建虚拟环境并安装项目依赖。
"""

import os
import subprocess
import platform
import argparse
from pathlib import Path

def main():
    """主函数：执行环境设置"""
    parser = argparse.ArgumentParser(description='为旅行时间预测项目设置环境')
    parser.add_argument('--no-venv', action='store_true', help='不创建虚拟环境，直接安装依赖')
    args = parser.parse_args()
    
    # 项目根目录
    root_dir = Path(__file__).resolve().parent
    
    # 要求确认
    print("此脚本将为旅行时间预测项目设置环境：")
    if not args.no_venv:
        print("1. 创建Python虚拟环境")
    print("2. 安装必要的依赖")
    
    confirm = input("继续？ (y/n): ")
    if confirm.lower() not in ['y', 'yes']:
        print("操作已取消")
        return
    
    # 创建虚拟环境
    if not args.no_venv:
        print("\n创建虚拟环境...")
        venv_dir = root_dir / 'venv'
        
        # 检查系统类型
        system = platform.system()
        if system == 'Windows':
            python_executable = 'python'
            activate_script = venv_dir / 'Scripts' / 'activate'
        else:
            python_executable = 'python3'
            activate_script = venv_dir / 'bin' / 'activate'
        
        # 创建虚拟环境
        try:
            subprocess.run([python_executable, '-m', 'venv', str(venv_dir)], check=True)
            print(f"虚拟环境已创建：{venv_dir}")
            
            # 打印激活命令
            if system == 'Windows':
                print(f"\n要激活虚拟环境，请运行：\n{venv_dir}\\Scripts\\activate")
            else:
                print(f"\n要激活虚拟环境，请运行：\nsource {venv_dir}/bin/activate")
                
        except subprocess.CalledProcessError as e:
            print(f"创建虚拟环境时出错：{e}")
            return
    
    # 安装依赖
    print("\n安装依赖...")
    requirements_file = root_dir / 'requirements.txt'
    
    if not requirements_file.exists():
        print(f"找不到依赖文件：{requirements_file}")
        return
    
    try:
        if not args.no_venv:
            # 在Windows上，使用不同的命令
            if system == 'Windows':
                pip_cmd = str(venv_dir / 'Scripts' / 'pip')
            else:
                pip_cmd = str(venv_dir / 'bin' / 'pip')
            
            subprocess.run([pip_cmd, 'install', '-r', str(requirements_file)], check=True)
        else:
            # 直接使用系统pip
            subprocess.run(['pip', 'install', '-r', str(requirements_file)], check=True)
            
        print("依赖安装完成")
        
    except subprocess.CalledProcessError as e:
        print(f"安装依赖时出错：{e}")
        return
    
    print("\n环境设置完成！你现在可以使用以下命令运行项目：")
    if not args.no_venv:
        if system == 'Windows':
            print(f"{venv_dir}\\Scripts\\activate")
        else:
            print(f"source {venv_dir}/bin/activate")
    print(f"python {root_dir}/run_pipeline.py --full")


if __name__ == '__main__':
    main() 