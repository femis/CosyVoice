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

# 使用清华大学的镜像源安装依赖到用户目录
RUN pip install --user -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host=pypi.tuna.tsinghua.edu.cn

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

# 复制已安装的 Python 包和项目文件
COPY --from=builder /root/.local /root/.local
COPY --from=builder /workspace /workspace

# 将用户目录添加到 PATH
ENV PATH=/root/.local/bin:$PATH

CMD ["python3", "api_run.py"]