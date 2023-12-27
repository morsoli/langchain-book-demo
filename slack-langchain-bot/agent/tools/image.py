import requests
import openai
import hashlib
from pathlib import Path
from langchain.tools import BaseTool
from typing import Optional
from langchain.callbacks.manager import CallbackManagerForToolRun

from utils import file_cache_dir
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# 工具描述
DESCRIPTION = """
当需要生成图像时有用。
输入：描述图像的详细提示
输出：最终的文件路径
"""

class GenerateImageTool(BaseTool):
    name = "GenerateImage"
    description = DESCRIPTION

    def _run(
        self, 
        description: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Path:
        url = self.generate_image(description)
        temp_file_filename = file_cache_dir / f"{self.md5_encode(url)}.jpg"
        # 执行下载
        with open(temp_file_filename, "wb") as f:
            response = requests.get(url)
            f.write(response.content)
        return temp_file_filename
    
    def generate_image(self, description: str)-> str:
        """使用图像生成API来生成图像"""
        response = openai.Image.create(
            model="dall-e-3",
            prompt=description,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        return image_url
    
    def md5_encode(self, url: str)-> str:
        md5_obj = hashlib.md5()
        # 更新哈希对象，注意将字符串编码为字节串
        md5_obj.update(url.encode('utf-8'))
        # 获取 MD5 编码后的十六进制字符串
        return md5_obj.hexdigest()