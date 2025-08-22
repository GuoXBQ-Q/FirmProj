import os
import re
import logging
import colorlog

# 创建一个带有颜色的日志格式
log_colors_config = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red'
}
class Logger:
    def __init__(self, name, level="DEBUG", use_stream=True, log_file=None, log_file_level=None):
        self.logger = logging.getLogger(name)
        self.set_level(level)
        # 设置日志格式
        # formatter = logging.Formatter('[%(levelname)s]: %(message)s')
        
        if not use_stream and not log_file:
            raise ValueError("Must specify either use_stream or log_file")

        if use_stream:
            formatter = colorlog.ColoredFormatter(
                        fmt='%(log_color)s{%(threadName)s} [%(levelname)s]: %(message)s',
                        log_colors=log_colors_config
                    )
            # 创建控制台handler，并设置级别为DEBUG
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.level)
            console_handler.setFormatter(formatter)
            # 添加handler到logger中
            self.logger.addHandler(console_handler)

        if log_file:
            # 如果指定了log文件，创建文件handler，并设置级别为DEBUG
            if not os.path.exists(log_file):
                # os.path.basename(log_file)
                os.system(f"touch {log_file}")
            formatter = logging.Formatter('{%(threadName)s} [%(levelname)s]: %(message)s')
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(log_file_level if log_file_level else self.level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        

    def set_level(self, level):
        if level == "DEBUG":
            self.level = logging.DEBUG
        elif level == "INFO":
            self.level = logging.INFO
        elif level == "WARNING":
            self.level = logging.WARNING
        elif level == "ERROR":
            self.level = logging.ERROR
        elif level == "CRITICAL":
            self.level = logging.CRITICAL
        else:
            raise ValueError("Invalid level: %s" % level)
        self.logger.setLevel(self.level)
    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)



def ensure_log_directory(log_dir):
    """确保日志目录存在"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

def get_latest_log_number(log_dir, phase ,process_dataset):
    """获取当前日志文件中的最大编号"""
    # 获取日志目录下的所有文件
    log_files = [f for f in os.listdir(log_dir) if f.startswith(f"{phase}_logger_{process_dataset}_") and f.endswith(".log")]
    
    if not log_files:
        return 0  # 如果没有日志文件，返回 0
    
    # 提取文件名中的编号
    max_number = 0
    pattern = re.compile(rf"{phase}_logger_{process_dataset}_(\d+)\.log")  # 匹配命名规则
    for file_name in log_files:
        match = pattern.match(file_name)
        if match:
            number = int(match.group(1))
            max_number = max(max_number, number)
    
    return max_number