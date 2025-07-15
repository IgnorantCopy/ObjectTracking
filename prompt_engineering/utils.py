import sys


path_to_key = ""
if sys.platform.startswith('linux'):
    path_to_key = "/home/nju-student/mkh/api_key"
elif sys.platform.startswith('win'):
    path_to_key = "C:/Users/27344/Desktop/api_key"


def get_api_key(type: str, filename=path_to_key) -> str:
    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith(type):
                return line.split(': ')[1][:-1]
    return ""
