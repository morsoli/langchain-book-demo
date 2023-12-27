import requests
from pathlib import Path
from typing import Optional, Tuple
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate

ROLE_PROMPT = f"""
从此刻起，你将扮演学习助教的角色。只回答自己了解到的事实，不了解的事实就礼貌的回应自己不知道。\n
如果别人询问无关的的问题，只回答'我只是一个由莫尔索创建的学习助教'
"""

human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", ROLE_PROMPT),
    ("human", human_template),
])


from libs.usage import UsageTracker
from utils import md5, get_text_from_whisper, format_dialog_text
from utils import index_cache_dir
from agent.agent_api import langchain_agent

class SlackContext:
    def __init__(self, event: dict, say, user: str, thread_ts: str):
        self.event = event
        self.say = say
        self.user = user
        self.thread_ts = thread_ts

class SlackAPIHandler:
    
    def __init__(self, slack_app):
        self.client = slack_app.client
        self.voice_extension_allowed = ['m4a', 'webm', 'mp3', 'wav']
        self.image_extension_allowed = ["png", "jpg", "jepg", "webp"]
        self.file_extension_allowed = ["pdf", "txt", "mdx", "md", "markdown"]
        self.max_file_size = 10 * 1024 * 1024
        self.usage = UsageTracker()
        self.chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    def process_event(self, event: dict, say, logger) -> None:
        user = event["user"]
        thread_ts = event["ts"]
        context = SlackContext(event, say, user, thread_ts)
        print(f"最新消息：{event}")
        
        if event.get("files"):
            self.handle_file_upload(context)
            file = event['files'][0]
            filetype = file["filetype"]
            file_md5_name = self.download_file(file, user)
            if filetype in self.voice_extension_allowed:
                voicemessage = get_text_from_whisper(file_md5_name)
                dialog_text = format_dialog_text(event["text"], voicemessage)
        else:
            dialog_text = format_dialog_text(event["text"])
            voicemessage = None
            
        self.process_conversation(context, dialog_text, voicemessage)

    def handle_file_upload(self, context: SlackContext) -> Tuple[Optional[str], Optional[str]]:
        try:
            file = context.event['files'][0]
            filetype = file["filetype"]
            say = context.say
            user = context.user
            thread_ts = context.thread_ts
            
            allowed_extension = self.voice_extension_allowed + self.image_extension_allowed + self.file_extension_allowed
            if filetype not in allowed_extension:
                say(f"<@{user}>, 当前只支持 {allowed_extension} 格式的文件", thread_ts=thread_ts)

            if file["size"] > self.max_file_size:
                say(f"<@{user}>, 文件大小超过限制 ({self.max_file_size / 1024 / 1024}MB)", thread_ts=thread_ts)
        except Exception as e:
            print(e)
        
    def download_file(self, file: dict, user: str) -> Optional[str]:
        url_private = file["url_private"]
        temp_file_path = index_cache_dir / user
        temp_file_path.mkdir(parents=True, exist_ok=True)
        temp_file_filename = temp_file_path / file["name"]
        # 执行下载
        with open(temp_file_filename, "wb") as f:
            response = requests.get(url_private, headers={"Authorization": "Bearer " + self.client.token})
            f.write(response.content)
        # 生成MD5名称
        filetype = file["filetype"]
        file_md5_name = self.generate_md5_name(temp_file_filename, filetype)
        return file_md5_name

    def generate_md5_name(self, temp_file_filename: Path, filetype: str) -> str:
        temp_file_md5 = md5(temp_file_filename)
        file_md5_name = index_cache_dir / (temp_file_md5 + '.' + filetype)
        if not file_md5_name.exists():
            temp_file_filename.rename(file_md5_name)
        return str(file_md5_name)

    def process_conversation(self, context: SlackContext, dialog_text: Optional[str], voicemessage: Optional[str] = None) -> None:
        gpt_response = self.chat_model.invoke(dialog_text).content
        if voicemessage is None:
            context.say(f'<@{context.user}>, {gpt_response}', thread_ts=context.thread_ts)
        else:
            voice_file_path = langchain_agent(context.user, f"语音重复下面内容 {gpt_response}")
            voice_file_path = "data/voice_cache/3269b614-28e9-4c97-a818-5c93ad93d2f2.mp3"
            self.client.files_upload_v2(file=voice_file_path, channel=context.event["channel"], thread_ts=context.thread_ts)
    
    def check_usage(self, context: SlackContext) -> bool:
        if not self.usage.exists(context.user):
            self.usage.add_user(context.user)
        if self.usage.usage_exceeded(context.user):
            self.send_messages()
            return False
        return True