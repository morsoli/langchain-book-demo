import re
import os
import hashlib
import openai
from typing import Optional
from pathlib import Path

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

