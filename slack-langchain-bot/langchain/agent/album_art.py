# 用于生成专辑封面的工具

import json
import logging

from langchain.agents import Tool
# 根据您的具体实现替换 Steamship 和 GenerateImageTool 的引用
# 从 langchain.agents 导入 Tool
# 假设 GenerateImageTool 在同一目录下

NAME = "生成专辑封面"

DESCRIPTION = """
在需要生成专辑封面时有用。
输入：需要封面的专辑描述
输出：生成的图像的 UUID
"""

class GenerateAlbumArtTool(Tool):
    """用于根据专辑描述生成专辑封面的工具。"""

    tool: GenerateImageTool

    def __init__(self):
        super().__init__(
            name=NAME,
            func=self.run,
            description=DESCRIPTION,
            tool=GenerateImageTool(),
        )

    @property
    def is_single_input(self) -> bool:
        """工具是否只接受单一输入。"""
        return True

    def run(self, prompt: str, **kwargs) -> str:
        """响应 LLM 提示。"""

        # 在这里，我们创建一个新的提示，基于提供给此工具的提示，
        # 但包括额外的术语。
        image_gen_prompt = f"专辑封面, 4k, 高清, 流行艺术, 专业, 高品质, 获奖, 格莱美, 白金唱片, {prompt}"

        # 然后我们返回包装的 GenerateImageTool 的结果，
        # 传递我们创建的新提示。
        return self.tool.run(image_gen_prompt)
