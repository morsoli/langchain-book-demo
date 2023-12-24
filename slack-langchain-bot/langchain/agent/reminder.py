# 用于安排提醒的工具
import logging
from typing import Callable

from langchain.agents import Tool
from pydantic import BaseModel, Field
from pytimeparse.timeparse import timeparse

class ToolRequest(BaseModel):
    @classmethod
    def get_json(cls):
        return {
            key: info["description"] for key, info in cls.schema()["properties"].items()
        }

class ReminderRequest(ToolRequest):
    """为 LLM 工具调用提供结构。"""

    after: str = Field(description="时间差")
    reminder: str = Field(description="要发送给用户的提醒消息")

# 示例
EXAMPLES = [
    ReminderRequest(after="15s", reminder="关灯"),
    ReminderRequest(after="60m", reminder="报税"),
    ReminderRequest(after="2h5m", reminder="提醒你的妻子晚餐的事情"),
]

NAME: str = "提醒"

EXAMPLES_STR = "\n".join([example.json() for example in EXAMPLES])
DESCRIPTION: str = f"""用于安排用户未来时间点的提醒。输入：时间差和提醒内容。请使用以下 JSON 格式作为输入：
{ReminderRequest.get_json()}。
            
示例：
{EXAMPLES_STR}""".replace("{", "{{").replace("}", "}}")

class RemindMe(Tool):
    """用于通过 Steamship 任务系统安排提醒的工具。"""

    invoke_later: Callable
    chat_id: str

    @property
    def is_single_input(self) -> bool:
        """工具是否仅接受单一输入。"""
        return True

    def __init__(self, invoke_later: Callable, chat_id: str):
        super().__init__(
            name="提醒",
            func=self.run,
            description=DESCRIPTION,
            invoke_later=invoke_later,
            chat_id=chat_id,
        )

    def run(self, prompt, **kwargs) -> str:
        """响应 LLM 提示。"""
        logging.info(f"[提醒我] 提示: {prompt}")
        if isinstance(prompt, dict):
            req = ReminderRequest.parse_obj(prompt)
        elif isinstance(prompt, str):
            prompt = prompt.replace("'", '"')
            req = ReminderRequest.parse_raw(prompt)
        else:
            return "工具失败。无法处理请求。抱歉。"

        self._schedule(req)
        return "这是输出内容"

    def _schedule(self, req: ReminderRequest) -> str:
        after_seconds = timeparse(req.after)
        logging.info(f"计划在 {after_seconds}s 后, 消息 {req.reminder}")

        self.invoke_later(
            delay_ms=after_seconds * 1_000,
            message=req.reminder,
            chat_id=self.chat_id,
        )

        logging.info(f"计划 {after_seconds * 1_000} 毫秒, 消息 {req.reminder}")

        return "您的提醒已被安排。"
