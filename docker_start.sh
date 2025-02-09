docker-compose -p voice up --build --force-recreate -d

#-p voice: 指定项目名称为 voice，这会影响容器和服务的命名。
#up: 启动定义在 docker-compose.yml 文件中的服务。
#--build: 强制重新构建镜像，即使镜像已经存在。
#--force-recreate: 强制重新创建容器，即使配置没有变化。
#-d: 在后台运行容器

docker build --progress=plain -t cosyvoice .

# 使用 docker-compose 启动服务
docker-compose -p voice up --force-recreate -d

# 编译不启动
docker build -t voice-cosy .