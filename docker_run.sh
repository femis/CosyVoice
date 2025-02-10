#!/bin/bash
echo "正在解压环境依赖..."

# 检查文件夹是否存在，如果存在则跳过解压
if [ -d "/opt/venv" ]; then
    echo "环境依赖已存在，跳过解压"
else
    tar -xzf /opt/venv.tar.gz -C /opt
    echo "正在解压完成"
fi

sleep 3

if [ -f "/opt/venv.tar.gz" ]; then
    rm -rf /opt/venv.tar.gz
fi


# 检查文件夹是否存在，如果存在则跳过解压
if [ -d "pretrained_models" ]; then
    echo "预训练模型已存在"
else
    wget http://host.docker.internal:8080/cosy/pretrained_models.tar.gz
    echo "正在解压预训练模型..."
    tar -xzf pretrained_models.tar.gz
    echo "正在解压完成"
fi

sleep 3

echo "正在删除预训练模型压缩包..."

if [ -f "pretrained_models.tar.gz" ]; then
    rm -rf pretrained_models.tar.gz
fi

echo

export PYTHONPATH=/opt/venv:$PYTHONPATH

echo "正在启动服务..."

python3 api_run.py
