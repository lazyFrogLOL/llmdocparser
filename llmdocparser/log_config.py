import logging.handlers
import os

# 确保日志目录存在
LOG_DIR = "../logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 配置日志文件路径
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# 配置日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 检查是否已经存在处理器，避免重复添加
if not logger.hasHandlers():
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建文件处理器
    file_handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.INFO)

    # 创建格式化器，并将其添加到处理器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 将处理器添加到记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)