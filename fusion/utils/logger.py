class Logger(object):
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.log_file = open(log_path, 'w')

    def log(self, message: str):
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()

    def close(self):
        self.log_file.close()