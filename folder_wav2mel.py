import os
import librosa
import numpy as np
import cv2

def audio_to_mel(audio_file, sr=16000, n_fft=2048, hop_length=256, n_mels=256, REF=None):
    """
    加载音频文件并计算对数梅尔频谱。
    """
    try:
        # 加载音频文件
        y, sr = librosa.load(audio_file, sr=sr)

        # 计算梅尔频谱
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
                                                  win_length=512)
        if REF is not None:
            # 转换为对数刻度
            log_mel_spec = librosa.amplitude_to_db(mel_spec, ref=REF)
        else:
            log_mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)

        # 防止静音音频导致全黄（clip 限制范围）
        log_mel_spec = np.clip(log_mel_spec, -80, 0)  # 限制 dB 范围，-80 到 0

        return log_mel_spec
    except Exception as e:
        print(f"处理文件 {audio_file} 时出错: {e}")
        return None


def save_mel_spectrogram(audio_file, output_dir, REF=400):
    """
    计算并保存 Mel 频谱图到指定的输出目录。
    """
    log_mel_spec = audio_to_mel(audio_file, REF=REF)
    if log_mel_spec is None:
        return

    # 归一化到 [0, 255] 范围
    mel_spec_uint8 = (log_mel_spec + 80) / 80 * 255  # 归一化到 0-255
    mel_spec_uint8 = mel_spec_uint8.astype(np.uint8)

    # 颜色映射
    mag_color = cv2.applyColorMap(mel_spec_uint8, cv2.COLORMAP_INFERNO)[::-1, :, :]

    # 构建输出文件路径
    base_filename = os.path.basename(audio_file)  # 获取原始文件名，例如 'audio.wav'
    output_filename = os.path.splitext(base_filename)[0] + '_mel.png'  # 例如 'audio_mel.png'
    output_path = os.path.join(output_dir, output_filename)

    # 保存图片
    cv2.imwrite(output_path, mag_color)
    print(f"已保存 Mel 频谱图: {output_path}")


def main():
    # # --- 1. 设置命令行参数解析 ---
    # parser = argparse.ArgumentParser(description="将WAV音频文件转换为梅尔频谱图。")
    # parser.add_argument('-i', '--input_dir', type=str, required=True,
    #                     help="包含WAV文件的输入文件夹路径。")
    # parser.add_argument('-o', '--output_dir', type=str, required=True,
    #                     help="用于保存输出PNG图像的文件夹路径。")
    # parser.add_argument('--ref', type=int, default=500,
    #                     help="用于dB转换的参考值 (REF)。")
    
    # args = parser.parse_args()
    input_dir = r"D:\VSCodeProjects\AlignSep\demo_audio\alignsep\wav"
    output_dir = r"D:\VSCodeProjects\AlignSep\demo_audio\alignsep\mel"
    ref = None

    # --- 2. 检查并创建输出文件夹 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    # --- 3. 遍历输入文件夹并处理文件 ---
    print(f"正在从 '{input_dir}' 读取文件...")
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_file_path = os.path.join(root, file)
                save_mel_spectrogram(audio_file_path, output_dir, REF=ref)

    print("\n处理完成！")


if __name__ == "__main__":
    main()