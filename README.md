# LiveTalking - 音频驱动视频生成系统

LiveTalking 是一个基于 Wav2Lip 的音频驱动视频生成系统，可以将音频文件转换为带有口型同步的视频。

## 功能特点

- 支持音频到视频的转换
- 支持自定义头像
- 支持自定义视频帧率
- 智能处理静音片段
- 自动音频视频合成
- 支持多种设备（CPU/CUDA/MPS）

## 环境要求

- Python 3.6+
- PyTorch
- OpenCV
- librosa
- soundfile
- tqdm
- ffmpeg（用于音频视频合成）

## 安装依赖

```bash
pip install torch opencv-python librosa soundfile tqdm
```

## 使用方法

### 基本使用

```python
from audio_to_video import process_audio_to_video

# 基本参数
audio_path = "sourcevoice.wav"  # 音频文件路径
avatar_id = "wav2lip256_avatar1"  # 头像ID
output_path = "output.mp4"  # 输出视频路径

# 处理音频到视频
process_audio_to_video(audio_path, avatar_id, output_path)
```

### 高级参数

```python
process_audio_to_video(
    audio_path="sourcevoice.wav",
    avatar_id="wav2lip256_avatar1",
    output_path="output.mp4",
    model_path="./models/wav2lip.pth",  # 模型路径
    fps=25  # 视频帧率
)
```

## 文件结构

```
LiveTalking/
├── audio_to_video.py    # 主程序文件
├── models/              # 模型文件夹
│   └── wav2lip.pth     # Wav2Lip模型文件
├── data/               # 数据文件夹
│   └── avatars/       # 头像数据
│       └── wav2lip256_avatar1/  # 示例头像
│           ├── full_imgs/      # 完整图像
│           ├── face_imgs/      # 面部图像
│           └── coords.pkl      # 坐标信息
└── README.md          # 说明文档
```

## 头像数据要求

每个头像文件夹应包含以下内容：
- `full_imgs/`: 完整图像文件夹
- `face_imgs/`: 面部图像文件夹
- `coords.pkl`: 面部坐标信息文件

## 注意事项

1. 确保音频文件格式为WAV格式
2. 确保模型文件存在于指定路径
3. 确保头像数据格式正确
4. 建议使用GPU进行加速处理
5. 如需音频视频合成，请确保系统已安装ffmpeg

## 输出文件

程序会生成两个文件：
1. `{audio_filename}_{output_filename}`: 原始视频文件
2. `{audio_filename}_{output_filename}_with_audio.mp4`: 带音频的视频文件

## 错误处理

- 如果ffmpeg未安装，程序会提示手动合成音频视频
- 如果文件已存在，程序会自动删除并重新生成
- 所有错误信息都会通过logger记录

## 许可证

请遵守相关开源协议使用本程序。

## 贡献

欢迎提交Issue和Pull Request来帮助改进这个项目。

[English](./README-EN.md) | 中文版   
实时交互流式数字人，实现音视频同步对话。基本可以达到商用效果
[wav2lip效果](https://www.bilibili.com/video/BV1scwBeyELA/) | [ernerf效果](https://www.bilibili.com/video/BV1G1421z73r/) | [musetalk效果](https://www.bilibili.com/video/BV1gm421N7vQ/)

## 为避免与3d数字人混淆，原项目metahuman-stream改名为livetalking，原有链接地址继续可用

## News
- 2024.12.8 完善多并发，显存不随并发数增加
- 2024.12.21 添加wav2lip、musetalk模型预热，解决第一次推理卡顿问题。感谢[@heimaojinzhangyz](https://github.com/heimaojinzhangyz)
- 2024.12.28 添加数字人模型Ultralight-Digital-Human。 感谢[@lijihua2017](https://github.com/lijihua2017)
- 2025.2.7 添加fish-speech tts
- 2025.2.21 添加wav2lip256开源模型 感谢@不蠢不蠢
- 2025.3.2 添加腾讯语音合成服务
- 2025.3.16 支持mac gpu推理，感谢[@GcsSloop](https://github.com/GcsSloop) 
- 2025.5.1 精简运行参数，ernerf模型移至git分支ernerf-rtmp

## Features
1. 支持多种数字人模型: ernerf、musetalk、wav2lip、Ultralight-Digital-Human
2. 支持声音克隆
3. 支持数字人说话被打断
4. 支持全身视频拼接
5. 支持rtmp和webrtc
6. 支持视频编排：不说话时播放自定义视频
7. 支持多并发

## 1. Installation

Tested on Ubuntu 20.04, Python3.10, Pytorch 1.12 and CUDA 11.3

### 1.1 Install dependency

```bash
conda create -n nerfstream python=3.10
conda activate nerfstream
#如果cuda版本不为11.3(运行nvidia-smi确认版本)，根据<https://pytorch.org/get-started/previous-versions/>安装对应版本的pytorch 
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
#如果需要训练ernerf模型，安装下面的库
# pip install "git+https://github.com/facebookresearch/pytorch3d.git"
# pip install tensorflow-gpu==2.8.0
# pip install --upgrade "protobuf<=3.20.1"
``` 
安装常见问题[FAQ](https://livetalking-doc.readthedocs.io/zh-cn/latest/faq.html)  
linux cuda环境搭建可以参考这篇文章 <https://zhuanlan.zhihu.com/p/674972886>  
视频连不上解决方法 <https://mp.weixin.qq.com/s/MVUkxxhV2cgMMHalphr2cg>


## 2. Quick Start
- 下载模型  
夸克云盘<https://pan.quark.cn/s/83a750323ef0>    
GoogleDriver <https://drive.google.com/drive/folders/1FOC_MD6wdogyyX_7V1d4NDIO7P9NlSAJ?usp=sharing>  
将wav2lip256.pth拷到本项目的models下, 重命名为wav2lip.pth;  
将wav2lip256_avatar1.tar.gz解压后整个文件夹拷到本项目的data/avatars下
- 运行  
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1  
用浏览器打开http://serverip:8010/webrtcapi.html , 先点'start',播放数字人视频；然后在文本框输入任意文字，提交。数字人播报该段文字  
<font color=red>服务端需要开放端口 tcp:8010; udp:1-65536 </font>  
如果需要商用高清wav2lip模型，[链接](https://livetalking-doc.readthedocs.io/zh-cn/latest/service.html#wav2lip) 

- 快速体验  
<https://www.compshare.cn/images-detail?ImageID=compshareImage-18tpjhhxoq3j&referral_code=3XW3852OBmnD089hMMrtuU&ytag=GPU_GitHub_livetalking1.3> 用该镜像创建实例即可运行成功

如果访问不了huggingface，在运行前
```
export HF_ENDPOINT=https://hf-mirror.com
``` 


## 3. More Usage
使用说明: <https://livetalking-doc.readthedocs.io/>
  
## 4. Docker Run  
不需要前面的安装，直接运行。
```
docker run --gpus all -it --network=host --rm registry.cn-beijing.aliyuncs.com/codewithgpu2/lipku-metahuman-stream:2K9qaMBu8v
```
代码在/root/metahuman-stream，先git pull拉一下最新代码，然后执行命令同第2、3步 

提供如下镜像
- autodl镜像: <https://www.codewithgpu.com/i/lipku/metahuman-stream/base>   
[autodl教程](https://livetalking-doc.readthedocs.io/en/latest/autodl/README.html)
- ucloud镜像: <https://www.compshare.cn/images-detail?ImageID=compshareImage-18tpjhhxoq3j&referral_code=3XW3852OBmnD089hMMrtuU&ytag=GPU_livetalking1.3>  
可以开放任意端口，不需要另外部署srs服务.  
[ucloud教程](https://livetalking-doc.readthedocs.io/en/latest/ucloud/ucloud.html) 


## 5. TODO
- [x] 添加chatgpt实现数字人对话
- [x] 声音克隆
- [x] 数字人静音时用一段视频代替
- [x] MuseTalk
- [x] Wav2Lip
- [x] Ultralight-Digital-Human

---
如果本项目对你有帮助，帮忙点个star。也欢迎感兴趣的朋友一起来完善该项目.
* 知识星球: https://t.zsxq.com/7NMyO 沉淀高质量常见问题、最佳实践经验、问题解答  
* 微信公众号：数字人技术  
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/l3ZibgueFiaeyfaiaLZGuMGQXnhLWxibpJUS2gfs8Dje6JuMY8zu2tVyU9n8Zx1yaNncvKHBMibX0ocehoITy5qQEZg/640?wxfrom=12&tp=wxpic&usePicPrefetch=1&wx_fmt=jpeg&amp;from=appmsg)  
