import librosa

def calculate_total_duration(txt_file):
    total_duration = 0.0

    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 分割每行内容
            parts = line.strip().split('|')
            audio_path = parts[0]  # 获取音频文件路径
            
            try:
                # 使用librosa加载音频文件
                y, sr = librosa.load(audio_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)  # 计算时长
                total_duration += duration
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")

    return total_duration

# 示例用法
txt_file = 'A5_finetuned_trainingout/gentle_girl/filelists/script.txt.cleaned.train'  # 替换为你的txt文件路径
total_duration = calculate_total_duration(txt_file)
print(f"Total duration: {total_duration:.2f} seconds, {total_duration / 60 :.2f} min  ")