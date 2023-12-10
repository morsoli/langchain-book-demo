import requests
import torch
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.tools import BaseTool
from PIL import Image
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

HF_ACCESS_TOEKN = os.getenv("HF_ACCESS_TOEKN")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

class ImageDescTool(BaseTool):
    name = "Image description"  # 工具名称
    description = "当你有一张图片的 URL 并想要获取这张图片的描述时，就可以使用这个工具, 它会生成一个简洁的说明文字来描述这张图片。"
    
    # 本地调用方式
    # 指定要使用的深度学习模型
    # hf_model = "Salesforce/blip-image-captioning-large"
    # # 如果GPU可用则使用GPU，否则使用CPU
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # processor = BlipProcessor.from_pretrained(hf_model)
    # # 初始化模型本身
    # model = BlipForConditionalGeneration.from_pretrained(hf_model).to(device)
    # def run(self, url: str):
    #     # 从URL下载图片并转换为PIL对象
    #     image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    #     # 对图像进行预处理
    #     inputs = processor(image, return_tensors="pt").to(device)
    #     # 生成描述
    #     out = model.generate(**inputs, max_new_tokens=20)
    #     # 获取描述
    #     caption = processor.decode(out[0], skip_special_tokens=True)
    #     return caption
    
    # 在线调用的方式
    def _run(self, url: str):
        headers = {"Authorization": f"Bearer {HF_ACCESS_TOEKN}"}
        data = requests.get(url, stream=True).raw
        # 在线使用
        model_api_url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
        response = requests.post(model_api_url, headers=headers, data=data)
        return response.json()[0].get("generated_text")

if __name__ == "__main__":
    # 创建工具实例并初始化agent
    tools = [ImageDescTool()]
    agent = initialize_agent(tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=3,
        handle_parsing_errors=True)
    # 运行agent
    img_url = "https://images.unsplash.com/photo-1598677997257-f8153318c049?q=80&w=1587&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    agent.run(f"这张图片里是什么？用中文回答\n{img_url}")
    