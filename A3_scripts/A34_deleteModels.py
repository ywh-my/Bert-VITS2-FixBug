import os
import re

def delete_files_less_than_N(directory, N):
    # 正则表达式匹配 <prefix>_<number>.pth 格式的文件
    pattern = re.compile(r'.*_(\d+)\.pth$')

    # 遍历目录中的文件
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # 获取文件中的数字
            file_number = int(match.group(1))
            # 如果文件中的数字小于 N，则删除文件
            if file_number < N or filename[0] != "G":
                file_path = os.path.join(directory, filename)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

# 示例使用
directory = 'A5_finetuned_trainingout/SSB0005_50/models'  # 目录A的路径
N = 8000  # 你想要删除的文件序号阈值
delete_files_less_than_N(directory, N)