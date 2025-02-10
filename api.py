import os
import sys
import argparse
import time
import requests
from fastapi import FastAPI, Form, HTTPException
import uuid
from flask import Flask, request, jsonify, send_file
import numpy as np
import torch
import torchaudio
import librosa
from pydub import AudioSegment
import logging
torch.cuda.set_per_process_memory_fraction(0.5)  # 使用50%的GPU显存
# 配置日志
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app = Flask(__name__)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
matcha_dir = os.path.join(ROOT_DIR, "third_party", "Matcha-TTS")
sys.path.insert(0, matcha_dir)

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
max_val = 0.8

# 定义全局变量
cosyvoice = None
sft_spk = None
prompt_sr = 16000
is_makeing = 0         #是否正在制作中
default_data = np.zeros(prompt_sr)  # 初始化为默认值

def initialize_cosyvoice(model_dir):
    global cosyvoice, sft_spk
    try:
        cosyvoice = CosyVoice2(model_dir)
        sft_spk = cosyvoice.list_available_spks()
        if len(sft_spk) == 0:
            sft_spk = ['']
        logger.info(f"初始化完成，可用说话人: {sft_spk}")
    except Exception as e:
        logger.error(f"初始化CosyVoice失败: {str(e)}", exc_info=True)
        cosyvoice = None
        sft_spk = None

def download_and_convert_audio(url: str) -> str:
    """下载并转换音频文件为wav格式"""
    logger.info(f"开始下载并转换音频，URL: {url}")
    # 从URL获取文件扩展名
    file_ext = os.path.splitext(url)[1] or '.mp3'
    logger.debug(f"获取文件扩展名: {file_ext}")
    # 下载文件
    temp_file = f"TEMP/{uuid.uuid4().hex}{file_ext}"
    try:
        logger.info(f"开始下载文件到临时路径: {temp_file}")
        response = requests.get(url)
        with open(temp_file, "wb") as f:
            f.write(response.content)
        logger.info(f"文件下载完成，大小: {os.path.getsize(temp_file)} bytes")

        # 转换格式
        wav_file = f"TEMP/{uuid.uuid4().hex}.wav"
        logger.info(f"开始音频格式转换，目标文件: {wav_file}")
        audio = AudioSegment.from_file(temp_file)
        audio.export(wav_file, format="wav")
        logger.info(f"音频转换完成，大小: {os.path.getsize(wav_file)} bytes")

        # 删除临时文件
        logger.debug(f"删除临时文件: {temp_file}")
        os.remove(temp_file)
        logger.info(f"音频处理完成，返回路径: {wav_file}")
        return wav_file
    except Exception as e:
        logger.error(f"音频处理失败: {str(e)}")
        # 清理可能存在的临时文件
        if 'temp_file' in locals() and os.path.exists(temp_file):
            logger.debug(f"清理临时文件: {temp_file}")
            os.remove(temp_file)
        if 'wav_file' in locals() and os.path.exists(wav_file):
            logger.debug(f"清理wav文件: {wav_file}")
            os.remove(wav_file)
        raise e

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    if cosyvoice is None:
        logger.error("cosyvoice 未初始化")
        raise HTTPException(status_code=500, detail="cosyvoice 未初始化")
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech

def generate_audio(tts_text, mode, sft_dropdown, prompt_text, prompt_wav_path, instruct_text, seed, stream, speed):
    """生成音频"""
    logger.info(f"开始生成音频，模式: {mode}")
    logger.debug(f"输入参数 - tts_text: {tts_text}, sft_dropdown: {sft_dropdown}, "
                f"prompt_text: {prompt_text}, prompt_wav_path: {prompt_wav_path}, "
                f"instruct_text: {instruct_text}, seed: {seed}, stream: {stream}, speed: {speed}")

    if cosyvoice is None:
        logger.error("cosyvoice 未初始化")
        raise HTTPException(status_code=500, detail="cosyvoice 未初始化")

    # 参数校验
    if mode not in inference_mode_list:
        logger.error(f"无效的模式: {mode}")
        raise ValueError(f"Invalid mode: {mode}")

    if mode == '自然语言控制':
        if not instruct_text:
            logger.error("缺少instruct文本")
            raise HTTPException(status_code=400, detail="请输入instruct文本")

    if mode in ['3s极速复刻', '跨语种复刻']:
        if not prompt_wav_path:
            logger.error("缺少prompt音频")
            raise HTTPException(status_code=400, detail="请提供prompt音频")
        if torchaudio.info(prompt_wav_path).sample_rate < prompt_sr:
            logger.error(f"prompt音频采样率低于{prompt_sr}")
            raise HTTPException(status_code=400, detail=f"prompt音频采样率低于{prompt_sr}")

    # 生成音频
    try:
        if mode == '预训练音色':
            logger.info('开始预训练音色模式推理')
            logger.debug(f'模型类型: {type(cosyvoice)}')
            logger.debug(f'输入文本: {tts_text}')
            logger.debug(f'说话人ID: {sft_dropdown}')
            set_all_random_seed(seed)
            inference_gen = cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed)
            logger.info('预训练音色推理生成器创建成功')
            for i in inference_gen:
                if i is None:
                    logger.error("模型返回空输出")
                    raise ValueError("Model returned None output")
                if 'tts_speech' not in i:
                    logger.error("模型输出缺少'tts_speech'字段")
                    raise ValueError("Model output missing 'tts_speech' key")
                logger.debug(f"生成音频数据，长度: {len(i['tts_speech'].numpy().flatten())}")
                yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
        elif mode == '3s极速复刻':
            logger.info('开始3s极速复刻模式推理')
            prompt_speech_16k = postprocess(load_wav(prompt_wav_path, prompt_sr))
            set_all_random_seed(seed)
            for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):
                logger.debug(f"生成音频数据，长度: {len(i['tts_speech'].numpy().flatten())}")
                yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
        elif mode == '跨语种复刻':
            logger.info('开始跨语种复刻模式推理')
            prompt_speech_16k = postprocess(load_wav(prompt_wav_path, prompt_sr))
            set_all_random_seed(seed)
            for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):
                logger.debug(f"生成音频数据，长度: {len(i['tts_speech'].numpy().flatten())}")
                yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
        else:
            logger.info('开始自然语言控制模式推理')
            prompt_speech_16k = postprocess(load_wav(prompt_wav_path, prompt_sr))
            set_all_random_seed(seed)
            for i in cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k, stream=stream, speed=speed):
                logger.debug(f"生成音频数据，长度: {len(i['tts_speech'].numpy().flatten())}")
                yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    except Exception as e:
        logger.error(f"音频生成失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"音频生成失败: {str(e)}")

#开始生成
@app.route("/tts", methods=['POST'])
def generate():
    global is_makeing
    data = request.get_json()
    is_makeing = 1
    start_time = time.time()
    """处理TTS请求"""
    logger.info("收到TTS请求")
    logger.debug(f"请求数据: {data}")
    # 从JSON获取参数
    tts_text = data.get("gen_text", "大家好，我是麦克阿瑟上校，听说你们都在引用我的名言。")
    language = data.get("language", "") #自然语言控制
    mode = "自然语言控制" if language else "3s极速复刻"
    sft_dropdown = "default"  # 固定值
    prompt_text = data.get("prompt_text", "大家好，我是麦克阿瑟上校，听说你们都在引用我的名言。")
    prompt_url = data.get("ref_audio_path", "https://vr-static.he29.com/public/case/rabbit/model-maikease.mp3")
    instruct_text = language  # 语言控制
    seed = data.get("seed", "0")  # 随机种子
    stream = False  # 固定值
    speed = data.get("speed_factor", 1.0)

    logger.info(f"处理参数 - tts_text: {tts_text}, prompt_text: {prompt_text} prompt_url: {prompt_url}, speed: {speed}")

    # 处理音频URL
    prompt_wav_path = None
    if prompt_url:
        try:
            logger.info(f"开始处理音频URL: {prompt_url}")
            prompt_wav_path = download_and_convert_audio(prompt_url)
            logger.info(f"音频处理完成，路径: {prompt_wav_path}")
        except Exception as e:
            logger.error(f"音频处理失败: {str(e)}")
            raise HTTPException(status_code=400, detail=f"音频处理失败: {str(e)}")

    # 生成音频
    output_file = f"TEMP/{uuid.uuid4().hex}.wav"
    try:
        logger.info(f"开始生成音频，输出文件: {output_file}")
        logger.info(f"------------------------------------------")
        logger.info(f"传入文案: {tts_text}")
        logger.info(f"参考文本: {prompt_text}")
        logger.info(f"参考音频: {prompt_wav_path}")
        logger.info(f"随机种子: {seed}")
        logger.info(f"------------------------------------------")
        audio_segments = []  # 用于存储所有音频片段
        for sr, audio_data in generate_audio(
            tts_text=tts_text,
            mode=mode,
            sft_dropdown=sft_dropdown,
            prompt_text=prompt_text,
            prompt_wav_path=prompt_wav_path,
            instruct_text=instruct_text,
            seed=int(seed),
            stream=stream,
            speed=speed
        ):
            # 使用与webui.py相同的音频处理方式
            audio_data = np.clip(audio_data, -1, 1)  # 限制音频范围
            audio_data = (audio_data * 32767).astype(np.int16)  # 转换为16位PCM格式
            audio_segments.append((sr, audio_data))  # 收集音频片段

        # 合并所有音频片段
        if audio_segments:
            combined_sr, combined_audio_data = audio_segments[0]
            for sr, audio_data in audio_segments[1:]:
                if sr != combined_sr:
                    raise ValueError("所有音频片段的采样率必须相同")
                combined_audio_data = np.concatenate((combined_audio_data, audio_data))
            torchaudio.save(output_file, torch.from_numpy(combined_audio_data).unsqueeze(0), combined_sr, format="mp3")
            logger.info(f"音频保存成功，路径: {output_file}")
        else:
            logger.error("没有生成任何音频片段")
            raise HTTPException(status_code=500, detail="没有生成任何音频片段")

        end_time = time.time()
        logger.info(f"制作请求完成; 耗时：{end_time - start_time}秒")
        return {
            "success": 1,
            "path": output_file,
            "time": end_time - start_time,
            "speed": speed
        }
    except Exception as e:
        logger.error(f"音频生成失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        is_makeing = 1
        logger.debug(f"执行完成!")

# 下载文件
@app.route('/download', methods=['GET'])
def download():
    # 从GET请求获取文件路径
    file_path = request.args.get('path', '')
    try:
        # 检查路径是否包含 TEMP
        if 'TEMP' in file_path:
            # 截取 TEMP 后面的部分
            relative_path = file_path.split('TEMP', 1)[1].lstrip(os.sep)
            # 构建完整的路径
            full_path = os.path.join(PROJECT_ROOT, 'TEMP', relative_path)
        else:
            # 直接使用传递的相对路径
            full_path = os.path.join(PROJECT_ROOT, 'TEMP', file_path)
        # 文件不存在就返回404
        if not os.path.exists(full_path):
            return jsonify({'error': '文件不存在'}), 404
        # 从本地读取文件并且输出
        return send_file(full_path, as_attachment=True)
    except Exception as e:
        logger.error(f"下载文件时发生异常: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 查询是否制作中;
@app.route('/', methods=['GET'])
def find():
    try:
        result = {
            'is_makeing': is_makeing
        }
        return jsonify(result)
    except Exception as e:
        logger.info(f"处理 / 请求时发生异常: {e}")  # 记录异常信息
        return jsonify({"error": "处理请求时发生内部错误"}), 500  # 返回错误响应

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6010)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B', help='')
    args = parser.parse_args()

    # 初始化全局变量
    initialize_cosyvoice(args.model_dir)

    # 检查初始化是否成功
    if cosyvoice is None:
        logger.error("cosyvoice 初始化失败，无法启动服务")
        sys.exit(1)
    else:
        logger.info("cosyvoice 初始化成功")
    # 创建必要的目录
    os.makedirs("TEMP", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # 启动 FastAPI 应用
    app.run(debug=False, host="0.0.0.0", port=args.port)
    logger.info(f"服务已启动: {args.port}")
