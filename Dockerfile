# 构建阶段
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

ARG VENV_NAME="cosyvoice"
ENV VENV=$VENV_NAME
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "--login", "-c"]

# 添加apt-get国内源并安装依赖
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update -y --fix-missing && \
    apt-get install -y git build-essential curl wget ffmpeg unzip git-lfs sox libsox-dev python3.10 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

# 安装依赖到集中目录并压缩
RUN mkdir -p /opt/venv && \
    pip install --target /opt/venv -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host=pypi.tuna.tsinghua.edu.cn && \
    cd /opt && \
    tar -czf venv.tar.gz venv && \
    rm -rf venv

COPY . .

# 运行阶段
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG VENV_NAME="cosyvoice"
ENV VENV=$VENV_NAME
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "--login", "-c"]

# 添加apt-get国内源并安装依赖
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update -y --fix-missing && \
    apt-get install -y wget ffmpeg sox libsox-dev python3.10 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 复制压缩的依赖和项目文件
COPY --from=builder /opt/venv.tar.gz /opt/
COPY --from=builder /workspace /workspace

EXPOSE 6010 6011

# 修改 CMD 指令以运行 docker_run.sh 脚本
CMD ["/workspace/docker_run.sh"]