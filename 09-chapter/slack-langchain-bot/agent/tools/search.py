import os
from langchain.tools import BaseTool
from typing import Optional
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.utilities.serpapi import SerpAPIWrapper

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

SERPAPI_API_KEY= os.environ.get("SERPAPI_API_KEY")

# 工具描述
DESCRIPTION = """
在需要回答有关最新新闻的问题时有用，只有在用户明确要求时才使用。
输入：查询意图
输出：最终的搜索结果
"""

class SearchTool(BaseTool):
    name = "Search"
    description = DESCRIPTION

    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """使用搜索API来执行搜索"""
        search_wrapper = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY, params={"engine": "baidu", "gl": "cn", "hl": "zh-cn"})
        return search_wrapper.run(query)
