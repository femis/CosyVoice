import subprocess

script = "api.py"
ports = [6010, 6011]  # 你可以根据需要更改这些端口号

# 启动子进程来运行每个脚本
processes = []
for port in ports:
    process = subprocess.Popen(f"python3 {script} --port={port}", shell=True, close_fds=True)
    processes.append(process)

# 等待所有子进程完成
for process in processes:
    process.wait()

print("所有脚本已运行完毕")