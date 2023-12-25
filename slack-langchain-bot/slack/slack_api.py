import os
import requests
from typing import Optional, Tuple
from slack_bolt import App, BoltResponse
from slack_bolt.error import BoltUnhandledRequestError

from libs.usage import UsageTracker
from utils import md5, get_text_from_whisper, get_voice_file_from_text, format_dialog_text
from utils import index_cache_dir

def init_slack_app() -> App:
    """初始化并配置 Slack Bot 应用"""
    # 创建一个App实例，配置token和signing_secret
    slack_app = App(
        token=os.environ.get("SLACK_TOKEN"),  # 从环境变量获取Slack的OAuth访问令牌
        signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),  # 从环境变量获取Slack的签名密钥
        raise_error_for_unhandled_request=True  # 对于未处理的请求抛出错误
    )
    # 定义错误处理器
    @slack_app.error
    def handle_errors(error):
        if isinstance(error, BoltUnhandledRequestError):  # 未处理的请求错误
            return BoltResponse(status=200, body="")
        else:
            return BoltResponse(status=500, body="出错了！")  # 其他错误
    
    slack_api_handler = SlackAPIHandler(slack_app.client)  # 创建一个处理Slack API事件的处理器

    # 定义消息事件处理函数
    @slack_app.event("message")
    def handle_message(event, say, logger):
        slack_api_handler.process_event(event, say, logger)  # 处理收到的消息事件

    return slack_app  # 返回配置好的Slack应用实例

class SlackContext:
    def __init__(self, event: dict, say, user: str, thread_ts: str):
        self.event = event
        self.say = say
        self.user = user
        self.thread_ts = thread_ts

class SlackAPIHandler:
    
    def __init__(self, slack_client):
        self.client = slack_client
        self.voice_extension_allowed = ['m4a', 'webm', 'mp3', 'wav']
        self.max_file_size = 10 * 1024 * 1024
        self.usage = UsageTracker()

    def process_event(self, event: dict, say) -> None:
        user = event["user"]
        thread_ts = event["ts"]
        context = SlackContext(event, say, user, thread_ts)

        file_md5_name, voicemessage = (None, None)
        if event.get('files'):
            file_md5_name, voicemessage = self.handle_file_upload(context)

        dialog_text = format_dialog_text(event["text"], voicemessage)
        self.process_conversation(context, dialog_text, voicemessage, file_md5_name)

    def handle_file_upload(self, context: SlackContext) -> Tuple[Optional[str], Optional[str]]:
        file = context.event['files'][0]
        filetype = file["filetype"]
        say = context.say
        user = context.user
        thread_ts = context.thread_ts

        if filetype != "pdf":
            say(f"<@{user}>, 当前只支持 PDF 格式的文件", thread_ts=thread_ts)
            return None, None

        if file["size"] > self.max_file_size:
            say(f"<@{user}>, 文件大小超过限制 ({self.max_file_size / 1024 / 1024}MB)", thread_ts=thread_ts)
            return None, None

        return self.process_file_upload(file, user)

    def process_file_upload(self, file: dict, user: str) -> Tuple[Optional[str], Optional[str]]:
        url_private = file["url_private"]
        filetype = file["filetype"]
        temp_file_path = index_cache_dir / user
        temp_file_path.mkdir(parents=True, exist_ok=True)
        temp_file_filename = temp_file_path / file["name"]
        with open(temp_file_filename, "wb") as f:
            response = requests.get(url_private, headers={"Authorization": "Bearer " + self.client.token})
            f.write(response.content)
            temp_file_md5 = md5(temp_file_filename)
            file_md5_name = index_cache_dir / (temp_file_md5 + '.' + filetype)
            if not file_md5_name.exists():
                temp_file_filename.rename(file_md5_name)
                if filetype in self.voice_extension_allowed:
                    voicemessage = get_text_from_whisper(file_md5_name)
            return file_md5_name, voicemessage

    def process_conversation(self, context: SlackContext, dialog_text: Optional[str], voicemessage: Optional[str] = None, file_md5_name: Optional[str] = None) -> None:
        gpt_response = f"langchain_process {dialog_text} {file_md5_name}"

        if voicemessage is None:
            context.say(f'<@{context.user}>, {gpt_response}', thread_ts=context.thread_ts)
        else:
            voice_file_path = get_voice_file_from_text(str(gpt_response))
            self.client.files_upload_v2(file=voice_file_path, channel=context.event["channel"], thread_ts=context.thread_ts)
    
    def check_usage(self, context: SlackContext) -> bool:
        if not self.usage.exists(context.user):
            self.usage.add_user(context.user)
        if self.usage.usage_exceeded(context.user):
            self.send_messages()
            return False
        return True