# 用于生成语音的工具
import json
import logging
from typing import Optional

from langchain.agents import Tool

# 工具名称
NAME = "生成语音"

# 工具描述
DESCRIPTION = (
    "用于根据文本提示生成语音。只有在用户明确要求语音输出时使用。"
    "使用此工具时，输入应为包含要说话内容的纯文本字符串。"
)

# 插件句柄
PLUGIN_HANDLE = "elevenlabs"


class GenerateSpeechTool(Tool):
    """用于根据文本提示生成语音的工具。"""

    voice_id: Optional[str] = "21m00Tcm4TlvDq8ikWAM"  # 使用的声音ID，默认为Rachel
    elevenlabs_api_key: Optional[str] = ""  # Elevenlabs的API密钥
    name: Optional[str] = NAME
    description: Optional[str] = DESCRIPTION

    def __init__(
        self,
        voice_id: Optional[str] = "21m00Tcm4TlvDq8ikWAM",
        elevenlabs_api_key: Optional[str] = "",
    ):
        super().__init__(
            name=NAME,
            func=self.run,
            description=DESCRIPTION,
            voice_id=voice_id,
            elevenlabs_api_key=elevenlabs_api_key,
        )

    @property
    def is_single_input(self) -> bool:
        """工具是否仅接受单一输入。"""
        return True

    def run(self, prompt: str, **kwargs) -> str:
        """响应LLM提示。"""
        logging.info(f"[{self.name}] {prompt}")

        # 这里替换为您自己的语音生成逻辑
        # 模拟一个语音生成任务
        try:
            # 假设生成了语音，并获取到了语音的ID
            audio_id = "模拟生成的语音ID"
            return audio_id
        except Exception as e:
            logging.error(f"[{self.name}] 工具无法生成语音: {e}")
            raise e  # 抛出异常
