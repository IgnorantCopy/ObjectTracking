from openai import OpenAI
import base64
import glob
import os
from tqdm import tqdm

import utils
from fusion.utils.visualize import plot_3d_trajectory


DATA_ROOT = "D:/DataSets/挑战杯_揭榜挂帅_CQ-08赛题_数据集/点迹"
CACHE_DIR = os.path.join(DATA_ROOT, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
NUM_CLASSES = 4

api_key = utils.get_api_key('openai')


def encode_image(image_path: str):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_cls(image_path: str) -> str:
    prompt = '''
    接下来我会给你一张针对低空'低慢小'目标的航迹图，请你根据下面给出的判断依据，帮我判断图中表示的是什么类别的目标（一共有四个类别：轻型旋翼无人机、
    小型旋翼无人机、鸟、空飘球）：
    1. 鸟类飞行轨迹随机性很强
    2. 空飘球飞行轨迹无随机性，随风移动，高度变化缓慢，呈平滑曲线，速度接近环境风速（通常小于10m/s）
    3. 无人机飞行轨迹随机性中等，多呈直线/折线
    四个类别分别用编号1、2、3、4表示，请只输出对应编号表示类别，也就是说，只要数字，不要其他文字。
    '''
    image_url = {
        "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
    }
    client = OpenAI(
        api_key=api_key,
    )
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_url}
            ]
        }],
        max_tokens=300,
    )
    return completion.choices[0].message.content


def main():
    point_files = glob.glob(os.path.join(DATA_ROOT, "PointTracks_*.txt"))
    totals = [0 for _ in range(NUM_CLASSES)]
    corrects = [0 for _ in range(NUM_CLASSES)]
    for point_file in tqdm(point_files, desc="Classification using VLM"):
        info = os.path.basename(point_file).split("_")
        batch_id = info[1]
        label = info[2]
        num_points = info[3]
        save_path = os.path.join(CACHE_DIR, f"Trajectory_{batch_id}_{label}_{num_points}.png")
        if not os.path.exists(save_path):
            plot_3d_trajectory(point_file, save_path=save_path)
        cls = get_cls(save_path)
        count = 0
        while not cls.isdigit() or int(cls) not in range(1, NUM_CLASSES+1):
            for i in range(NUM_CLASSES):
                if cls.find(str(i+1)) != -1:
                    cls = i+1
                    break
            else:
                count += 1
                if count > 3:
                    print(f"Cannot recognize class for {point_file}")
                    break
                cls = get_cls(save_path)
        cls = int(cls)
        if cls == int(label):
            corrects[cls-1] += 1
        totals[cls-1] += 1
    for i in range(NUM_CLASSES):
        print(f"Class {i+1}: {corrects[i]}/{totals[i]} ({corrects[i]/totals[i]:.2%})")
    print(f"Overall accuracy: {sum(corrects)/sum(totals):.2%}")


if __name__ == '__main__':
    main()