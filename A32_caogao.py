import logging

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,  # 设置日志等级为 INFO 及以上
    format='%(asctime)s - %(levelname)s - %(message)s'  # 设置日志格式
)

# 示例日志
logging.info("This is an info message.")