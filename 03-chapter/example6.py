import os
import io
import requests
from tqdm import tqdm
from pydantic import Field
from typing import List, Mapping, Optional, Any
from langchain.llms.base import LLM
from gpt4all import GPT4All
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class CustomLLM(LLM):
    """
    一个自定义的LLM类，用于集成gpt4all模型

    参数：

    model_folder_path: (str) 存放模型的文件夹路径
    model_name: (str) 要使用的模型名称（<模型名称>.bin）
    allow_download: (bool) 是否允许下载模型

    backend: (str) 模型的后端（支持的后端：llama/gptj）
    n_threads: (str) 要使用的线程数
    n_predict: (str) 要生成的最大token数
    temp: (str) 用于采样的温度
    top_p: (float) 用于采样的top-p值
    top_k: (float) 用于采样的top k值
    """
    # 以下是类属性的定义
    model_folder_path: str = Field(None, alias='model_folder_path')
    model_name: str = Field(None, alias='model_name')
    allow_download: bool = Field(None, alias='allow_download')

    # 所有可选参数

    backend:        Optional[str]   = 'llama'
    temp:           Optional[float] = 0.7
    top_p:          Optional[float] = 0.1
    top_k:          Optional[int]   = 40
    n_batch:        Optional[int]   = 8
    n_threads:      Optional[int]   = 4
    n_predict:      Optional[int]   = 256

    # 初始化模型实例
    gpt4_model_instance:Any = None 

    def __init__(self, model_folder_path, model_name, allow_download, **kwargs):
        super(CustomLLM, self).__init__()
        # 类构造函数的实现
        self.model_folder_path: str = model_folder_path
        self.model_name = model_name
        self.allow_download = allow_download
        
        # 触发自动下载
        self.auto_download()

        # 创建GPT4All模型实例
        self.gpt4_model_instance = GPT4All(
            model_name=self.model_name,
            model_path=self.model_folder_path,
        )

    def auto_download(self) -> None:
        """
        此方法将会下载模型到指定路径, 参考https://python.langchain.com/docs/integrations/llms/gpt4all
        """
        # 检查模型名称是否包含.bin后缀
        model_name = (
            f"{self.model_name}.bin"
            if not self.model_name.endswith(".bin")
            else self.model_name
        )

        download_path = os.path.join(self.model_folder_path, model_name)

        if not os.path.exists(download_path):
            if self.allow_download:
                # 向URL发送GET请求下载文件
                # 因为文件较大，所以边下载边流式传输
                try:
                    url = f'https://gpt4all.io/models/{model_name}'

                    response = requests.get(url, stream=True)
                    # 以二进制模式打开文件，并写入响应内容的块
                    with open(download_path, 'wb') as f:
                        for chunk in tqdm(response.iter_content(chunk_size=8912)):
                            if chunk: f.write(chunk)
                
                except Exception as e:
                    print(f"=> 下载失败。错误: {e}")
                    return
                
                print(f"=> 模型: {self.model_name} 已成功下载 🥳")
            
            else:
                print(
                    f"模型: {self.model_name} 不存在于 {self.model_folder_path}",
                    "请通过设置 allow_download = True 下载模型")
                
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        返回一个字典类型, 包含LLM的唯一标识
        """
        return {
            'model_name' : self.model_name,
            'model_path' : self.model_folder_path,
            **self._get_model_default_parameters
        }
    
    @property
    def _llm_type(self) -> str:
        return 'gpt4all'
    
    def _call(
            self, 
            prompt: str, stop: Optional[List[str]] = None, 
            **kwargs) -> str:
        """
        重写基类方法, 根据用户输入的prompt来响应用户, 返回字符串      
        """
        
        params = {
            **self._get_model_default_parameters, 
            **kwargs
        }

        with self.gpt4_model_instance.chat_session():
            response_generator = self.gpt4_model_instance.generate(prompt, **params)

            if params['streaming']:
                response = io.StringIO()
                for token in response_generator:
                    print(token, end='', flush=True)
                    response.write(token)
                response_message = response.getvalue()
                response.close()
                return response_message
        return response_generator
    
if __name__ == "__main__":
    llm = CustomLLM(model_folder_path= ("./models/"), model_name="ggml-gpt4all-l13b-snoozy.bin", allow_download=True)
    print(llm("讲一个笑话"))