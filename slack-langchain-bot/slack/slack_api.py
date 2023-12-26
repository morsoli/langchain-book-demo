import requests
from pathlib import Path
from typing import Optional, Tuple
from libs.usage import UsageTracker
from utils import md5, get_text_from_whisper, get_voice_file_from_text, format_dialog_text
from utils import index_cache_dir

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
        self.max_file_size = 10 * 1024 * 1024
        self.usage = UsageTracker()

    def process_event(self, event: dict, say, logger) -> None:
        user = event["user"]
        thread_ts = event["ts"]
        context = SlackContext(event, say, user, thread_ts)
        self.handle_file_upload(context)
        print(f"最新消息：{event['text']}")
        
        file = event['files'][0]
        filetype = file["filetype"]
        file_md5_name = self.download_file(file, user)
        if filetype in self.voice_extension_allowed:
            voicemessage = get_text_from_whisper(file_md5_name)
            dialog_text = format_dialog_text(event["text"], voicemessage)
        else:
            dialog_text = format_dialog_text(event["text"])
        self.process_conversation(context, dialog_text, voicemessage, file_md5_name)

    def handle_file_upload(self, context: SlackContext) -> Tuple[Optional[str], Optional[str]]:
        file = context.event['files'][0]
        filetype = file["filetype"]
        say = context.say
        user = context.user
        thread_ts = context.thread_ts

        # if filetype != "pdf":
        #     say(f"<@{user}>, 当前只支持 PDF 格式的文件", thread_ts=thread_ts)

        if file["size"] > self.max_file_size:
            say(f"<@{user}>, 文件大小超过限制 ({self.max_file_size / 1024 / 1024}MB)", thread_ts=thread_ts)
        
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