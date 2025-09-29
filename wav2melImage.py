import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf # 用于创建演示文件

def convert_wav_to_mel_spectrogram(wav_path, save_path):
    """
    将 WAV 文件转换为梅尔频谱图并保存为图像。

    参数:
    wav_path (str): 输入的 WAV 文件路径。
    save_path (str): 保存梅尔频谱图图像的路径 (例如 'output.png')。
    """
    try:
        # 1. 加载 WAV 文件
        y, sr = librosa.load(wav_path, sr=None)

        # 2. 计算梅尔频谱图
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)

        # 3. 将功率谱转换为分贝 (dB)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # 4. 创建图像并保存
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(log_mel_spectrogram, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
        
        # 添加颜色条、标题和标签
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram: {os.path.basename(wav_path)}')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        
        plt.tight_layout()

        # 5. 保存图像
        plt.savefig(save_path)
        
        # 6. 关闭图像，释放内存
        plt.close()

        print(f"成功将 {wav_path} 转换为梅尔频谱图并保存至 {save_path}")

    except Exception as e:
        print(f"处理文件 {wav_path} 时出错: {e}")

# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 步骤 1: 设置你的输入 WAV 文件路径 ---
    input_wav_path = r"D:\VSCodeProjects\AlignSep\demo_audio\target_wav\00069.wav"  # <--- 替换成你的 WAV 文件路径

    # --- 步骤 2: 自定义完整的输出图像路径和文件名 ---
    # 你可以指定任何文件夹和任何文件名。
    # 示例 (Windows): "C:/Users/YourUser/Desktop/my_spectrogram.png"
    # 示例 (Linux/Mac): "results/audio_analysis/spectrogram_01.png"
    output_image_path = r"D:\VSCodeProjects\AlignSep\demo_audio\target_mel\00069.png" # <--- 在这里设置你想要的完整保存路径

    # -------------------------------------------------------------

    # 检查输入文件是否存在 (如果不存在，则创建一个用于演示的虚拟文件)
    if not os.path.exists(input_wav_path):
        print(f"警告: 输入文件不存在 -> {input_wav_path}")
        print("正在创建一个用于演示的虚拟 WAV 文件 'your_audio.wav'...")
        sr_test = 22050
        duration = 5
        frequency = 440
        t = np.linspace(0., duration, int(sr_test * duration))
        amplitude = np.iinfo(np.int16).max * 0.5
        data = amplitude * np.sin(2. * np.pi * frequency * t)
        sf.write(input_wav_path, data.astype(np.int16), sr_test)
        print("虚拟文件创建成功。")

    # 从你设置的输出路径中提取目录部分
    output_dir = os.path.dirname(output_image_path)

    # 如果路径中包含文件夹 (例如 "custom_output/"), 则确保这个文件夹存在
    # 如果 output_dir 为空字符串 (表示保存在当前目录), 则不会创建文件夹
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已自动创建目录: {output_dir}")

    # 调用函数进行转换
    convert_wav_to_mel_spectrogram(input_wav_path, output_image_path)