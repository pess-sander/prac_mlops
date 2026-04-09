import logging
import os

class Logger:
    def __init__(self, config):
        log_file = config['logger']['log_file']
        log_level = config['logger']['log_level']
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        self.logger = logging.getLogger('MLOps')
        self.logger.setLevel(getattr(logging, log_level.upper()))

        formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger