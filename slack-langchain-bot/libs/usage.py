"""
这段代码的主要目的是通过 UsageTracker 类来跟踪和限制用户的消息使用情况，
并通过 RateLimiter 类来限制用户在给定时间段内的请求频率。
使用了 Pydantic 的 BaseModel 来定义 UsageEntry，这为消息使用情况提供了结构化的表示。代
码中充分考虑了类型注解，以增强代码的可读性和可维护性。
"""

from typing import Optional, Dict, List
from pydantic import BaseModel
from cache import KeyValueStore
import time


class UsageEntry(BaseModel):
    message_count: int = 0  # 用户已发送的消息数量
    message_limit: int = 0  # 用户的消息限额


class RateLimiter:
    def __init__(self, limit: int = 10, period: int = 3600):
        """初始化一个请求频率限制器。

        Args:
            limit (int): 时间段内允许的最大请求次数。
            period (int): 时间段长度（秒）。
        """
        self.limit = limit
        self.period = period
        self.users: Dict[str, List[float]] = {}  # 存储用户请求时间戳的字典

    def allow_request(self, user_id: str) -> bool:
        """判断给定用户的请求是否允许。

        Args:
            user_id (str): 用户ID。

        Returns:
            bool: 如果允许请求则为True，否则为False。
        """
        now = time.time()
        user_requests = self.users.get(user_id, [])
        user_requests = [req for req in user_requests if req > now - self.period]
        if len(user_requests) < self.limit:
            user_requests.append(now)
            self.users[user_id] = user_requests
            return True
        return False


class UsageTracker:
    def __init__(self, n_free_messages: Optional[int] = 0):
        """初始化一个用于跟踪用户消息使用情况的跟踪器。

        Args:
            n_free_messages (Optional[int]): 用户可免费使用的消息数量。
        """
        self.kv_store = KeyValueStore(store_identifier="usage_tracking")
        self.n_free_messages = n_free_messages
        self.rate_limiter = RateLimiter()

    def get_usage(self, chat_id: str) -> UsageEntry:
        """获取指定用户的消息使用情况。

        Args:
            chat_id (str): 用户的聊天ID。

        Returns:
            UsageEntry: 用户的消息使用情况。
        """
        if not self.user_exists(chat_id):
            self.add_user(chat_id)
        return UsageEntry(**self.kv_store.get(chat_id))

    def set_usage(self, chat_id: str, usage: UsageEntry) -> None:
        """设置指定用户的消息使用情况。

        Args:
            chat_id (str): 用户的聊天ID。
            usage (UsageEntry): 用户的消息使用情况。
        """
        self.kv_store.set(chat_id, usage.dict())

    def is_usage_exceeded(self, chat_id: str) -> bool:
        """检查指定用户是否超出其消息使用限额。

        Args:
            chat_id (str): 用户的聊天ID。

        Returns:
            bool: 如果超出限额则为True，否则为False。
        """
        usage_entry = self.kv_store.get(chat_id)
        return (
            usage_entry["message_limit"] > 0 and
            usage_entry["message_count"] >= usage_entry["message_limit"]
        )

    def add_user(self, chat_id: str) -> None:
        """为新用户添加消息使用情况记录。

        Args:
            chat_id (str): 用户的聊天ID。
        """
        self.set_usage(chat_id, UsageEntry(message_limit=self.n_free_messages))

    def user_exists(self, chat_id: str) -> bool:
        """检查指定用户是否已有消息使用情况记录。

        Args:
            chat_id (str): 用户的聊天ID。

        Returns:
            bool: 如果用户已存在则为True，否则为False。
        """
        return self.kv_store.get(chat_id) is not None

    def increase_message_count(self, chat_id: str, n_messages: Optional[int] = 1) -> int:
        """增加用户的消息计数。

        Args:
            chat_id (str): 用户的聊天ID。
            n_messages (Optional[int]): 要增加的消息数量，默认为1。

        Returns:
            int: 用户更新后的消息计数。

        Raises:
            Exception: 如果用户超出了频率限制。
        """
        if self.rate_limiter.allow_request(chat_id):
            usage_entry = self.get_usage(chat_id)
            usage_entry.message_count += n_messages
            self.set_usage(chat_id, usage_entry)
            return usage_entry.message_count
        else:
            raise Exception("超出频率限制")

    def increase_message_limit(self, chat_id: str, n_messages: int) -> None:
        """增加用户的消息限额。

        Args:
            chat_id (str): 用户的聊天ID。
            n_messages (int): 要增加的消息限额。
        """
        usage_entry = self.get_usage(chat_id)
        usage_entry.message_limit += n_messages
        self.set_usage(chat_id, usage_entry)
