import os
import random
import shutil

def copy_random_wav_files(src_dir, dest_dir, N):
    # 确保目标目录存在
    os.makedirs(dest_dir, exist_ok=True)

    # 获取所有.wav文件
    wav_files = [f for f in os.listdir(src_dir) if f.endswith('.wav')]
    
    # 随机选择N个文件
    selected_files = random.sample(wav_files, min(N, len(wav_files)))

    # 复制选定的文件到目标目录
    for file in selected_files:
        shutil.copy(os.path.join(src_dir, file), os.path.join(dest_dir, file))

# 示例用法
N = 50
copy_random_wav_files('/data/data-aishell3/train/wav/SSB0273', f'A2_prepared_audios/SSB0273_{N}', N=N)