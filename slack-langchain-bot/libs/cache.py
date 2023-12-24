"""
在这个实现中，我们使用 diskcache.Cache 来创建一个持久化的键值存储。
每个方法 (get, set, delete, items, reset) 对应于 diskcache API 提供的功能。
我们也添加了一个 close 方法来关闭缓存并在不再需要时释放资源。
这个 KeyValueStore 类可以用于在文件系统上持久化地存储和检索键值对数据。
这对于需要快速访问且数据量较大的应用场景非常有用。
由于 diskcache 将数据存储在磁盘上，因此即使在程序重启之后，数据依然可以被访问。
"""

import diskcache
from typing import Any, Optional, Tuple, List

class KeyValueStore:
    cache: diskcache.Cache

    def __init__(self, cache_dir: str) -> None:
        """创建一个新的 KeyValueStore 实例。

        Args:
            cache_dir (str): 存储缓存数据的目录路径。
        """
        self.cache = diskcache.Cache(cache_dir)

    def get(self, key: str) -> Optional[Any]:
        """获取键 `key` 对应的值。

        Args:
            key (str): 键名。

        Returns:
            Optional[Any]: 键对应的值，如果不存在则为 None。
        """
        return self.cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """设置键值对 (key, value)。

        Args:
            key (str): 键名。
            value (Any): 值。
        """
        self.cache.set(key, value)

    def delete(self, key: str) -> bool:
        """删除键 `key` 对应的条目。

        Args:
            key (str): 键名。

        Returns:
            bool: 如果键被成功删除返回 True，否则返回 False。
        """
        return self.cache.pop(key, None) is not None

    def items(self) -> List[Tuple[str, Any]]:
        """返回所有键值对作为 (key, value) 元组的列表。

        Returns:
            List[Tuple[str, Any]]: 包含所有键值对的列表。
        """
        return list(self.cache.items())

    def reset(self) -> None:
        """删除所有键值对。"""
        self.cache.clear()

    def close(self) -> None:
        """关闭缓存并释放资源。"""
        self.cache.close()
