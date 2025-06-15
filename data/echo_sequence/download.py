import requests
import os


def download_file(url, save_dir):
    """
    下载单个文件
    :param url: 下载链接
    :param save_dir: 保存目录
    """
    try:
        file_name = url.split('&fileName=')[-1]
        save_path = os.path.join(save_dir, file_name)

        if os.path.exists(save_path):
            print(f"文件 {file_name} 已存在")
            return

        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"下载完成：{file_name}")
    except requests.exceptions.RequestException as e:
        print(f"下载失败：{url}，错误：{e}")
    except Exception as e:
        print(f"处理错误：{url}，错误：{e}")


def main():
    # configs
    file_path = "720626420979597312.txt"
    save_dir = "./dataset"

    os.makedirs(save_dir, exist_ok=True)

    with open(file_path, 'r', encoding='utf-8') as f:
        links = [line.strip() for line in f if line.strip()]

    for url in links:
        download_file(url, save_dir)


if __name__ == "__main__":
    main()