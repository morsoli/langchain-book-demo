# 用于生成图像的工具
import json
import logging
from langchain.agents import Tool

# 工具名称
NAME = "生成图像"

# 工具描述
DESCRIPTION = """
当需要生成图像时有用。
输入：描述图像的详细提示
输出：生成图像的 UUID
"""

# 插件句柄
PLUGIN_HANDLE = "stable-diffusion"


class GenerateImageTool(Tool):
    """用于从文本提示生成图像的工具。"""

    def __init__(self):
        super().__init__(name=NAME, func=self.run, description=DESCRIPTION)

    @property
    def is_single_input(self) -> bool:
        """该工具是否只接受单一输入。"""
        return True

    def run(self, prompt: str, **kwargs) -> str:
        """响应 LLM 提示。"""

        # 使用日志记录提示信息
        logging.info(f"[{self.name}] {prompt}")
        if not isinstance(prompt, str):
            prompt = json.dumps(prompt)

        # 这里替换为您自己的图像生成逻辑
        # 模拟一个图像生成任务
        try:
            # 假设生成了一张图像，并获取到了图像的 ID
            image_id = "模拟生成的图像ID"
            return image_id
        except Exception as e:
            logging.error(f"[{self.name}] 工具无法生成图像: {e}")
            raise e  # 抛出异常
