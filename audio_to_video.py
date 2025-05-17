import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
import numpy as np
import cv2
import glob
import pickle
from wav2lip.models import Wav2Lip
from logger import logger
import librosa
import soundfile as sf
from tqdm import tqdm
import subprocess

device = "cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")

def __mirror_index(size, index):
    """计算镜像索引，使帧序列在循环时更加自然
    
    Args:
        size: 帧序列的总长度
        index: 当前索引
        
    Returns:
        镜像后的索引
    """
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1

def load_model(model_path):
    """加载Wav2Lip模型"""
    model = Wav2Lip()
    logger.info(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    model = model.to(device)
    return model.eval()

def load_avatar(avatar_id):
    """加载头像数据"""
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    
    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)

    return frame_list_cycle, face_list_cycle, coord_list_cycle

def read_imgs(img_list):
    """读取图片列表"""
    frames = []
    logger.info('Reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def get_mel(audio_path, fps=25):
    """从音频文件提取mel特征"""
    # 加载音频并设置采样率
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # 音频预处理
    # 使用基于最大值的标准化
    max_value = np.max(np.abs(audio))
    if max_value > 0:
        audio = audio / max_value
    # 应用增益
    gain = 4.0
    audio = audio*gain
    
    # 应用预加重滤波器
    pre_emphasis = 0.97
    emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
    
    # 计算梅尔频谱图参数
    n_fft = 4096  # FFT窗口大小
    hop_length = int(sr / fps)  # 根据fps计算每帧对应的音频样本数
    win_length = int(sr * 0.03)  # 5ms窗口
    n_mels = 80  # 梅尔滤波器组数量
    
    # 提取梅尔频谱图
    mel = librosa.feature.melspectrogram(
        y=emphasized_audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        power=1.0,
        fmin=0,
        fmax=8000
    )
    
    # 转换为对数刻度
    mel = np.log(np.clip(mel, 1e-5, None))
    
    # 转置以匹配模型期望的输入格式
    mel = np.transpose(mel, (1, 0))
    
    return mel, audio, sr

def is_silence(audio_segment, threshold=0.01):
    """检测音频段是否为静音
    
    Args:
        audio_segment: 音频数据
        threshold: 静音阈值
        
    Returns:
        bool: 是否为静音
    """
    return np.mean(np.abs(audio_segment)) < threshold

def process_audio_to_video(audio_path, avatar_id, output_path, model_path="./models/wav2lip.pth", fps=25):
    """处理音频到视频的转换
    
    Args:
        audio_path: 音频文件路径
        avatar_id: 头像ID
        output_path: 输出视频路径
        model_path: 模型路径
        fps: 视频帧率，默认为25fps
    """
    # 从音频路径中提取文件名（不含扩展名）
    audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
    
    # 构建新的输出路径
    output_dir = os.path.dirname(output_path)
    output_filename = os.path.basename(output_path)
    new_output_path = os.path.join(output_dir, f"{audio_filename}_{output_filename}")
    
    # 检查并清理已存在的文件
    temp_video_path = new_output_path.replace('.mp4', '_temp.mp4')
    output_with_audio = new_output_path.replace('.mp4', '_with_audio.mp4')
    
    # 删除已存在的文件
    for file_path in [new_output_path, temp_video_path, output_with_audio]:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Removed existing file: {file_path}")
            except Exception as e:
                logger.error(f"Error removing file {file_path}: {str(e)}")
                return
    
    # 加载模型和头像
    model = load_model(model_path)
    frame_list_cycle, face_list_cycle, coord_list_cycle = load_avatar(avatar_id)
    
    # 获取音频特征和音频长度
    mel, audio, sr = get_mel(audio_path, fps)
    audio_duration = len(audio) / sr  # 音频时长（秒）
    
    # 设置视频参数
    total_frames = int(audio_duration * fps)  # 根据音频时长计算所需的总帧数
    batch_size = 32
    
    # 确保mel特征的长度与视频帧数匹配
    if len(mel) < total_frames:
        # 如果mel特征长度不足，通过插值扩展
        mel = np.interp(
            np.linspace(0, len(mel)-1, total_frames),
            np.arange(len(mel)),
            mel
        )
    elif len(mel) > total_frames:
        # 如果mel特征长度过长，截取所需部分
        mel = mel[:total_frames]
    
    # 创建视频写入器
    height, width = frame_list_cycle[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    # 处理每一帧
    for i in range(0, total_frames, batch_size):
        # 准备批次数据
        current_batch_size = min(batch_size, total_frames - i)
        mel_batch = mel[i:i+current_batch_size]
        if mel_batch.shape[0] < batch_size:
            mel_batch = np.pad(mel_batch, ((0, batch_size - mel_batch.shape[0]), (0, 0)))
        
        # 检查当前批次是否全为静音
        is_all_silence = True
        for j in range(current_batch_size):
            if not is_silence(mel_batch[j]):
                is_all_silence = False
                break
        
        if is_all_silence:
            # 静音帧处理：直接使用原始帧
            for j in range(current_batch_size):
                if i + j >= total_frames:
                    break
                idx = __mirror_index(len(frame_list_cycle), i + j)
                combine_frame = frame_list_cycle[idx].copy()
                out.write(combine_frame)
        else:
            # 非静音帧处理：使用模型生成
            img_batch = []
            for j in range(batch_size):
                idx = __mirror_index(len(face_list_cycle), i + j)
                face = face_list_cycle[idx]
                img_batch.append(face)
            
            img_batch = np.asarray(img_batch)
            img_masked = img_batch.copy()
            img_masked[:, face.shape[0]//2:] = 0
            
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            
            # 调整mel特征的维度以匹配模型要求
            mel_batch = mel_batch.reshape(mel_batch.shape[0], 1, -1, 1)
            
            # 确保特征图大小足够
            target_height = 80
            target_width = 16
            
            if mel_batch.shape[2] < target_height:
                mel_batch_resized = np.zeros((mel_batch.shape[0], 1, target_height, 1))
                for b in range(mel_batch.shape[0]):
                    mel_batch_resized[b, 0, :, 0] = np.interp(
                        np.linspace(0, 1, target_height),
                        np.linspace(0, 1, mel_batch.shape[2]),
                        mel_batch[b, 0, :, 0]
                    )
                mel_batch = mel_batch_resized
            
            mel_batch = np.repeat(mel_batch, target_width, axis=3)
            
            # 转换为tensor
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(mel_batch).to(device)
            
            # 推理
            with torch.no_grad():
                pred = model(mel_batch, img_batch)
            
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            
            # 合成最终帧
            for j, res_frame in enumerate(pred):
                if i + j >= total_frames:
                    break
                    
                idx = __mirror_index(len(frame_list_cycle), i + j)
                bbox = coord_list_cycle[idx]
                combine_frame = frame_list_cycle[idx].copy()
                
                y1, y2, x1, x2 = bbox
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                combine_frame[y1:y2, x1:x2] = res_frame
                
                out.write(combine_frame)
    
    out.release()
    
    # 保存原始视频
    os.rename(temp_video_path, new_output_path)
    logger.info(f"Original video saved to {new_output_path}")
    
    # 尝试使用ffmpeg合成音频和视频
    try:
        # 检查ffmpeg是否可用
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("ffmpeg not found, please install ffmpeg to enable audio-video synthesis")
            return
        
        # 构建ffmpeg命令
        command = [
            'ffmpeg', '-y',
            '-i', new_output_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            output_with_audio
        ]
        
        # 执行命令
        subprocess.run(command, check=True)
        logger.info(f"Video with audio saved to {output_with_audio}")
        
    except Exception as e:
        logger.error(f"Error in audio-video synthesis: {str(e)}")
        logger.info("Please use video editing software to manually add audio")
        logger.info(f"Audio file: {audio_path}")
        logger.info(f"Video file: {new_output_path}")

if __name__ == "__main__":
    # 示例使用
    audio_path = "sourcevoice.wav"
    avatar_id = "wav2lip256_avatar1"
    output_path = "output.mp4"
    process_audio_to_video(audio_path, avatar_id, output_path) 