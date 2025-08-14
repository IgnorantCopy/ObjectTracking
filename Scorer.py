import os
import pandas as pd 
import re

def get_txt_files(folder_path):
    """获取文件夹下所有txt文件路径"""
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"路径 {folder_path} 不是有效文件夹")
    
    txt_files = [
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) 
        if f.lower().endswith('.txt')
    ]
    return txt_files

def process_single_file(file_path, track_id_col = 1, model_col=-1, delimiter=',', ground_truth=None):
    """
    处理单个txt文件，提取航迹数据，计算首次正确识别的周期数（全局行号）
    参数：
        file_path (str): txt文件路径
        track_id_col (int): 航迹ID所在列的索引（从0开始）
        model_col (int): 模型识别结果所在列的索引（默认-1，即最后一列）
        delimiter (str): 列分隔符（默认逗号）
        ground_truth: 外部正确标签
    返回：
        周期 time(行号，而非时间)  sorted_track_data：按照行号排列，字典，包含批次和标签
    """
    # -------------------- 步骤1：读取文件并记录数据行号 --------------------
    try:
        # 读取txt文件（首行为表头，行号从0开始；数据行号从1开始计数）
        df = pd.read_csv(file_path, sep=delimiter, header=0, encoding='gbk')
    except Exception as e:
        raise RuntimeError(f"读取文件 {file_path} 失败: {str(e)}")
    
    total_rows = df.shape[0]  # 总数据行数（不含表头）
    if total_rows == 0:
        return {}  # 空文件无数据
    
    # -------------------- 步骤2：提取关键列数据并记录行号 --------------------
    # 遍历数据行（行号从1开始计数，对应表头后的第一行）
    track_data = {}
    for row_idx, (pid, ml) in enumerate(zip(
        df.iloc[:, track_id_col].astype(str),  # 航迹ID（字符串类型）
        df.iloc[:, model_col].astype(int)                  # 模型标签
    ), start=1):  # 数据行号从1开始（表头是第0行）
        # 仅保留模型标签非空的行（根据实际需求调整）
        if pd.notna(ml):
            track_data[row_idx] = {
                "行号": row_idx,
                "航迹ID": pid,
                "模型标签": ml
            }
    
    # -------------------- 步骤3：计算每个航迹的首次正确识别周期数 --------------------
    time = len(track_data)
    sorted_track_data = sorted(track_data.items(), key=lambda item: item[1]["行号"])
    for col, item in sorted_track_data :
        if item["模型标签"] == ground_truth:
            time = item["行号"]
            break
    
    return time, sorted_track_data

folder_path = "./results/航迹"
txt_files = get_txt_files(folder_path)
times = []
effective_lines = 0
for file in txt_files:
    ground_truth = int(re.match(r"Tracks_(\d+)_(\d+)_(\d+).txt", os.path.basename(file)).group(2))
    effective_point = 0
    time, sorted_track_data = process_single_file(file, ground_truth=ground_truth)
    times.append(time)
    for item in sorted_track_data :
        if item[1]["模型标签"] == ground_truth:
            effective_point = effective_point + 1
    if effective_point >= len(sorted_track_data) * 0.9:
        effective_lines = effective_lines + 1

P1 = effective_lines / len(txt_files) * 25

average_time = sum(times) / len(times)

print("P1:", P1)
print("平均周期:", average_time)
