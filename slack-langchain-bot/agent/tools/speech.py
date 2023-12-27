import os
import uuid
import random
import logging
from typing import Optional
from pathlib import Path
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from langdetect import detect
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, ResultReason, CancellationReason, SpeechSynthesisOutputFormat
from azure.cognitiveservices.speech.audio import AudioOutputConfig

from utils import voice_cache_dir
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 获取环境变量中的 API 密钥
SPEECH_KEY = os.environ.get("SPEECH_KEY")
SPEECH_REGION = os.environ.get("SPEECH_REGION")

# 语言代码到声音名称的映射字典
lang_code_voice_map = {
    "zh": ["zh-CN-XiaoxiaoNeural", "zh-CN-XiaohanNeural", "zh-CN-YunxiNeural", "zh-CN-YunyangNeural"],
    "en": ["en-US-JennyNeural", "en-US-RogerNeural", "en-IN-NeerjaNeural", "en-IN-PrabhatNeural", "en-AU-AnnetteNeural", "en-AU-CarlyNeural", "en-GB-AbbiNeural", "en-GB-AlfieNeural"],
}

DESCRIPTION = """
用于根据文本生成语音，只有在用户明确要求语音输出时使用。
输入：应为包含要说话内容的纯文本字符串
输出：最终的文件路径
"""


class GenerateVoiceTool(BaseTool):
    name = "GenerateVoice"
    description = DESCRIPTION

    def _run(
        self, 
        text: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Path:
        """使用文本到语音API来生成语音"""
        return self.generate_voice(text)
        

    # 检测文本的语言代码
    def detect_language(self, text: str) -> str:
        try:
            return detect(text).split("-")[0]
        except Exception as e:
            logging.error(e)
            return "zh"
            
        # 将文本转换为SSML格式
    def convert_to_ssml(self, text: str, voice_name: Optional[str] = None) -> str:
        lang_code = self.detect_language(text)
        voice_name = voice_name or random.choice(lang_code_voice_map.get(lang_code, lang_code_voice_map['zh']))
        ssml = f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-CN">'
        ssml += f'<voice name="{voice_name}">{text}</voice>'
        ssml += '</speak>'
        return ssml


    # 将文本转换为语音文件
    def generate_voice(self, text: str, voice_name: Optional[str] = None) -> Path:
        speech_config = SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
        speech_config.set_speech_synthesis_output_format(SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
        speech_config.speech_synthesis_language = "zh-CN"
        file_name = f"{voice_cache_dir}/{uuid.uuid4()}.mp3"
        file_config = AudioOutputConfig(filename=file_name)
        synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=file_config)
        ssml = self.convert_to_ssml(text, voice_name)
        result = synthesizer.speak_ssml_async(ssml).get()
        if result.reason == ResultReason.SynthesizingAudioCompleted:
            logging.info(f"Speech synthesized for text [{text}], and the audio was saved to [{file_name}]")
        elif result.reason == ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logging.error(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == CancellationReason.Error:
                logging.error(f"Error details: {cancellation_details.error_details}")
        return file_name
