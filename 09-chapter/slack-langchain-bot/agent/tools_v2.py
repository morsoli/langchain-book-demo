import os
import uuid
import random
import logging
import openai
import hashlib
import requests
from typing import Optional
from pathlib import Path
from langdetect import detect
from langchain_core.tools import tool
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, ResultReason, CancellationReason, SpeechSynthesisOutputFormat
from azure.cognitiveservices.speech.audio import AudioOutputConfig

from tools.utils import voice_cache_dir
from tools.utils import file_cache_dir


# 获取环境变量中的 API 密钥
SPEECH_KEY = os.environ.get("SPEECH_KEY")
SPEECH_REGION = os.environ.get("SPEECH_REGION")

# 语言代码到声音名称的映射字典
lang_code_voice_map = {
    "zh": ["zh-CN-XiaoxiaoNeural", "zh-CN-XiaohanNeural", "zh-CN-YunxiNeural", "zh-CN-YunyangNeural"],
    "en": ["en-US-JennyNeural", "en-US-RogerNeural", "en-IN-NeerjaNeural", "en-IN-PrabhatNeural", "en-AU-AnnetteNeural", "en-AU-CarlyNeural", "en-GB-AbbiNeural", "en-GB-AlfieNeural"],
}


@tool
def generate_image(description: str) -> Path:
    """使用图像生成API来生成图像"""
    response = openai.Image.create(
        model="dall-e-3",
        prompt=description,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    temp_file_filename = file_cache_dir / f"{md5_encode(image_url)}.jpg"
    # 执行下载
    with open(temp_file_filename, "wb") as f:
        response = requests.get(image_url)
        f.write(response.content)
    return temp_file_filename

@tool
def generate_voice(text: str, voice_name: Optional[str] = None) -> Path:
    """将文本转换为语音文件"""
    # return "语音文件路径为https://langchain-ai.github.io/langgraph/concepts/memory/"
    speech_config = SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.set_speech_synthesis_output_format(SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
    speech_config.speech_synthesis_language = "zh-CN"
    file_name = f"{voice_cache_dir}/{uuid.uuid4()}.mp3"
    file_config = AudioOutputConfig(filename=file_name)
    synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=file_config)
    ssml = convert_to_ssml(text, voice_name)
    result = synthesizer.speak_ssml_async(ssml).get()
    if result.reason == ResultReason.SynthesizingAudioCompleted:
        logging.info(f"Speech synthesized for text [{text}], and the audio was saved to [{file_name}]")
    elif result.reason == ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        logging.error(f"Speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == CancellationReason.Error:
            logging.error(f"Error details: {cancellation_details.error_details}")
    return file_name


# 检测文本的语言代码
def detect_language(text: str) -> str:
    try:
        return detect(text).split("-")[0]
    except Exception as e:
        logging.error(e)
        return "zh"
        
# 将文本转换为SSML格式
def convert_to_ssml(text: str, voice_name: Optional[str] = None) -> str:
    lang_code = detect_language(text)
    voice_name = voice_name or random.choice(lang_code_voice_map.get(lang_code, lang_code_voice_map['zh']))
    ssml = f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-CN">'
    ssml += f'<voice name="{voice_name}">{text}</voice>'
    ssml += '</speak>'
    return ssml

def md5_encode(url: str)-> str:
    md5_obj = hashlib.md5()
    # 更新哈希对象，注意将字符串编码为字节串
    md5_obj.update(url.encode('utf-8'))
    # 获取 MD5 编码后的十六进制字符串
    return md5_obj.hexdigest()