#!/bin/bash

# 解压之前压缩的文件

echo "解压之前压缩的文件..."

#新增判断, 若是已解压, 则跳过, 否则解压和删除
if [ -d "/root/.local/lib/python3.10" ]; then
  echo "已解压, 跳过解压"
else
  tar -xzf /root/.local/lib/python3.10.tar.gz -C /root/.local/lib/
  echo "解压完成，删除压缩文件"
  # 删除压缩文件
  rm /root/.local/lib/python3.10.tar.gz
fi


# 等待一秒钟
sleep 1

echo "正在启动服务..."

python3 api_run.py


