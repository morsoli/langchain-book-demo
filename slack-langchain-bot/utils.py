import re
import os
import hashlib
import openai
import random
import logging
import uuid
from langdetect import detect
from typing import Optional
from pathlib import Path
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, ResultReason, CancellationReason, SpeechSynthesisOutputFormat
from azure.cognitiveservices.speech.audio import AudioOutputConfig

# 获取环境变量中的 API 密钥
SPEECH_KEY = os.environ.get("SPEECH_KEY")
SPEECH_REGION = os.environ.get("SPEECH_REGION")

# 语言代码到声音名称的映射字典
lang_code_voice_map = {
    "zh": ["zh-CN-XiaoxiaoNeural", "zh-CN-XiaohanNeural", "zh-CN-YunxiNeural", "zh-CN-YunyangNeural"],
    "en": ["en-US-JennyNeural", "en-US-RogerNeural", "en-IN-NeerjaNeural", "en-IN-PrabhatNeural", "en-AU-AnnetteNeural", "en-AU-CarlyNeural", "en-GB-AbbiNeural", "en-GB-AlfieNeural"],
}

# 创建缓存目录
def create_cache_dir(dir_path: Path) -> None:
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path
        
# 设置缓存目录
voice_cache_dir = create_cache_dir(Path("./data/voice_cache/"))
file_cache_dir = create_cache_dir(Path("./data/file_cache/"))
web_cache_dir = create_cache_dir(Path("./data/web_cache/"))
index_cache_dir = create_cache_dir(Path("./data/index_cache/"))

# 计算文件的 MD5 值
def md5(file_path: Path) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# 检测文本的语言代码
def detect_language(text: str) -> str:
    try:
        return detect(text).split("-")[0]
    except Exception as e:
        logging.error(e)
        return "zh"

# 使用 Whisper 从语音文件中获取文本
def get_text_from_whisper(voice_file_path: Path) -> str:
    with open(voice_file_path, "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)
    return transcript.text

# 格式化对话文本
def format_dialog_text(text: str, voicemessage: Optional[str] = None) -> str:
    if text or voicemessage:
        text = insert_space(text)
        return text + ("\n" + voicemessage if voicemessage else "")
    else:
        return None

# 在中英文之间插入空格
def insert_space(text: str) -> str:
    text = re.sub(r"([a-zA-Z])([\u4e00-\u9fa5])", r"\1 \2", text)
    text = re.sub(r"([\u4e00-\u9fa5])([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"(\d)([\u4e00-\u9fa5])", r"\1 \2", text)
    text = re.sub(r"([\u4e00-\u9fa5])(\d)", r"\1 \2", text)
    text = re.sub(r"([\W_])([\u4e00-\u9fa5])", r"\1 \2", text)
    text = re.sub(r"([\u4e00-\u9fa5])([\W_])", r"\1 \2", text)
    return text.replace("  ", " ")

# 将文本转换为SSML格式
def convert_to_ssml(text: str, voice_name: Optional[str] = None) -> str:
    lang_code = detect_language(text)
    voice_name = voice_name or random.choice(lang_code_voice_map.get(lang_code, lang_code_voice_map['zh']))
    ssml = f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-CN">'
    ssml += f'<voice name="{voice_name}">{text}</voice>'
    ssml += '</speak>'
    return ssml


# 将文本转换为语音文件
def get_voice_file_from_text(text: str, voice_name: Optional[str] = None) -> Path:
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

