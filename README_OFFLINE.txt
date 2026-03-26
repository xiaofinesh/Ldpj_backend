Ldpj_backend 离线安装包
========================================
版本: v2.1
生成日期: 2026-02-25

本安装包包含 Ldpj_backend 系统的源代码及离线部署所需的 Python 依赖包。
适用于无法连接互联网的 Linux 工控机环境。

目录结构
----------------------------------------
- core/, configs/, ... : 源代码目录
- deploy/
  - offline_packages/  : 包含所有必需的 Python wheel 包 (.whl)
  - offline_install.sh : 自动化安装脚本
  - 离线部署操作手册.md : 详细部署文档
- requirements.txt     : 依赖列表
- README.md            : 项目说明

快速安装指南
----------------------------------------
1. 将本安装包传输至目标工控机 (Linux x86_64)。
   如果是一个压缩包 (Ldpj_backend_deploy.tar.gz)，请先解压：
   $ tar -xzvf Ldpj_backend_deploy.tar.gz
   $ cd Ldpj_backend

2. 确保系统已安装 Python 3.11+ 和 git (可选)。
   检查命令: python3 --version

3. (可选) 安装系统级依赖
   如果 snap7 连接失败，可能需要安装 libsnap7-dev。
   本包不包含 .deb 文件，请自行准备或确认系统已安装。

4. 执行一键安装脚本 (需要 root 权限)
   $ sudo bash deploy/offline_install.sh

   该脚本将自动：
   - 创建虚拟环境 (.venv)
   - 离线安装所有 Python 依赖
   - 部署项目到 /opt/ldpj_backend
   - 配置 systemd 服务 (ldpj_backend.service)

5. 启动服务
   $ sudo systemctl start ldpj_backend
   $ sudo systemctl status ldpj_backend

故障排查
----------------------------------------
- 如果报错 "command not found: python3"，请先安装 Python 3.11。
- 如果报错 "snap7 library not found"，请安装 snap7 系统库。
- 查看日志: tail -f /opt/ldpj_backend/logs/stdout.log

更多详细信息请参阅 `deploy/离线部署操作手册.md`。
