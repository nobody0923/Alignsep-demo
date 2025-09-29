import torchaudio
import numpy as np
import torch
import os
import shutil

duration = 8  # The target duration of the audio segments in seconds.
sampling_rate = 16000  # The sampling rate for saving the audio files.

def read_wav_file(filename):
    # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
    waveform, sr = torchaudio.load(filename)

    waveform, random_start = random_segment_wav(waveform, target_length=int(sr * duration))

    # waveform = resample(waveform, sr)
    # random_start = int(random_start * (sampling_rate / sr))

    waveform = waveform.numpy()[0, ...]

    waveform = normalize_wav(waveform)

    waveform = waveform[None, ...]
    waveform = pad_wav(waveform, target_length=int(sr * duration))
    return waveform, random_start

def normalize_wav(waveform):
    waveform = waveform - np.mean(waveform)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform * 0.5  # Manually limit the maximum amplitude into 0.5

def random_segment_wav(waveform, target_length):
    waveform_length = waveform.shape[-1]
    assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

    # Too short
    if (waveform_length - target_length) <= 0:
        return waveform, 0

    # random_start = int(
    #     random_uniform(0, waveform_length - target_length)
    # )
    random_start = 0
    return waveform[:, random_start: random_start + target_length], random_start

def random_uniform(self, start, end):
    val = torch.rand(1).item()
    return start + (end - start) * val

def pad_wav(waveform, target_length):
    waveform_length = waveform.shape[-1]
    assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

    if waveform_length == target_length:
        return waveform

    # Pad
    temp_wav = np.zeros((1, target_length), dtype=np.float32)

    temp_wav[:, 0: 0 + waveform_length] = waveform
    return temp_wav

def get_audio_mixed(waveform, noise_waveform):
    noise_waveform = noise_waveform[0][:len(waveform)]
    # waveform, noise_waveform = pad_arrays(waveform, noise_waveform)

    snr = 0

    # Avoid division by zero if noise is silent
    noise_power = np.mean(noise_waveform ** 2)
    if noise_power < 1e-10:
        # If noise is silent, just return the original waveform
        mixed_waveform = waveform
    else:
        source_power = np.mean(waveform ** 2)
        
        # Calculate scaling factor for the noise to achieve the target SNR
        desired_noise_power = source_power / (10 ** (snr / 10))
        scaling_factor = np.sqrt(desired_noise_power / noise_power)
        noise_waveform = noise_waveform * scaling_factor

        # Mix the audio
        mixed_waveform = waveform + noise_waveform

    # Normalize the mixture to prevent clipping if its amplitude exceeds 1.0
    max_value = np.max(np.abs(mixed_waveform))
    if max_value > 1.0:
        mixed_waveform *= 0.9 / max_value

    return torch.from_numpy(mixed_waveform.reshape(1, -1))

if __name__ == "__main__":
    # 1. Define output directory
    output_dir = r"D:\VSCodeProjects\AlignSep\demo_audio"
    os.makedirs(output_dir, exist_ok=True)

    # --- Create dummy audio files for demonstration ---
    # This part ensures the script is runnable out-of-the-box.
    # In a real scenario, you would replace these paths with your own files.
    target_file = r'D:\System\Desktop\dataset\vggsound_clean\noise_wav\000020.wav'
    noise_file = r'D:\System\Desktop\dataset\small_hardset\target_wav\000132.wav'
    cavp_file = r'D:\System\Desktop\dataset\vggsound_clean\target_cavp_4fps\000020.npz'
    video_file = r'D:\System\Desktop\dataset\vggsound_clean\target_video\000020.mp4'
    output_filename = "00002"
    
    if not os.path.exists(target_file):
        print(f"Creating dummy file for demonstration: {target_file}")
        t = np.linspace(0., duration, int(sampling_rate * duration), endpoint=False)
        data = 0.5 * np.sin(2. * np.pi * 440. * t) # A4 note
        torchaudio.save(target_file, torch.from_numpy(data).float().unsqueeze(0), sampling_rate)

    if not os.path.exists(noise_file):
        print(f"Creating dummy file for demonstration: {noise_file}")
        data = np.random.uniform(-0.5, 0.5, sampling_rate * duration) # White noise
        torchaudio.save(noise_file, torch.from_numpy(data).float().unsqueeze(0), sampling_rate)
    # --- End of dummy file creation ---

    # 2. Specify input WAV file paths
    target_wav_path = target_file
    noise_wav_path = noise_file

    print(f"\nProcessing target: {target_wav_path}")
    print(f"Processing noise:  {noise_wav_path}")

    # 3. Read and process the audio files using your function
    target_wav_np, _ = read_wav_file(target_wav_path) # returns numpy array (1, T)
    noise_wav_np, _ = read_wav_file(noise_wav_path)   # returns numpy array (1, T)
    
    # 4. Mix the target and noise waveforms
    # get_audio_mixed expects the target waveform to be 1D (squeezed)
    mixed_wav_tensor = get_audio_mixed(target_wav_np[0], noise_wav_np) # returns torch tensor (1, T)

    # 5. Define output file paths
    target_output_path = os.path.join(output_dir, "target_wav", f"{output_filename}.wav")
    noise_output_path = os.path.join(output_dir, "noise_wav", f"{output_filename}.wav")
    mixed_output_path = os.path.join(output_dir, "mixed_wav", f"{output_filename}.wav")
    # Per the request, this file is a copy of the processed target waveform
    target_cavp_output_path = os.path.join(output_dir, "target_cavp", f"{output_filename}.npz")
    target_video_output_path = os.path.join(output_dir, "target_video", f"{output_filename}.mp4")

    # 6. Save the resulting audio files
    # Convert numpy arrays to tensors before saving
    target_wav_tensor = torch.from_numpy(target_wav_np)
    noise_wav_tensor = torch.from_numpy(noise_wav_np)

    torchaudio.save(target_output_path, target_wav_tensor.float(), sampling_rate)
    print(f"Saved processed target audio to: {target_output_path}")

    torchaudio.save(noise_output_path, noise_wav_tensor.float(), sampling_rate)
    print(f"Saved processed noise audio to: {noise_output_path}")

    torchaudio.save(mixed_output_path, mixed_wav_tensor.float(), sampling_rate)
    print(f"Saved mixed audio to: {mixed_output_path}")
    
    shutil.copy2(cavp_file, target_cavp_output_path)
    shutil.copy2(video_file, target_video_output_path)
    