import os
import logging
import requests
from pathlib import Path
import concurrent.futures
from slack_bolt import App, BoltResponse
from slack_bolt.error import BoltUnhandledRequestError
from slack.slash_command import register_slack_slash_commands
from utils import md5, get_text_from_whisper, get_voice_file_from_text, format_dialog_text

def init_slack_app() -> App:
    """初始化并配置 Slack Bot 应用"""
    slack_app = App(
        token=os.environ.get("SLACK_TOKEN"),
        signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
        raise_error_for_unhandled_request=True
    )
    register_slack_slash_commands(slack_app)

    @slack_app.error
    def handle_errors(error):
        if isinstance(error, BoltUnhandledRequestError):
            return BoltResponse(status=200, body="")
        else:
            return BoltResponse(status=500, body="出错了！")
    
    slack_api_handler = SlackAPIHandler(slack_app.client)

    @slack_app.event("message")
    def handle_message(event, say, logger):
        slack_api_handler.process_event(event, say, logger)

    return slack_app


class SlackAPIHandler:
    MAX_THREAD_MESSAGE_HISTORY = 10  # 设置线程消息历史的最大长度
    
    def __init__(self, slack_client, index_cache_dir="./index_cache"):
        self.client = slack_client
        self.allowed_filetypes = ['epub', 'pdf', 'text', 'docx', 'markdown', 'm4a', 'webm', 'mp3', 'wav', 'md', 'mdx']
        self.voice_extension_allowed = ['m4a', 'webm', 'mp3', 'wav']
        self.max_file_size = 50 * 1024 * 1024
        self.index_cache_dir = Path(index_cache_dir)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)
        self.thread_message_history = {}

    def process_event(self, event, say, logger):
        user = event["user"]
        thread_ts = event["ts"]
        parent_thread_ts = event.get("thread_ts", thread_ts)
        file_md5_name, voicemessage = (None, None)

        if event.get('files'):
            file_md5_name, voicemessage = self.handle_file_upload(event, say, user, thread_ts)

        dialog_text = self.extract_and_format_dialog(event, voicemessage) if "text" in event or voicemessage else None
        self.update_thread_history(parent_thread_ts, dialog_text, self.extract_urls_from_event(event), file_md5_name)

        self.process_conversation(parent_thread_ts, event, say, user, thread_ts, logger, dialog_text, voicemessage, file_md5_name)

    def handle_file_upload(self, event, say, user, thread_ts):
        file = event['files'][0]
        filetype = file["filetype"]

        if filetype not in self.allowed_filetypes:
            say(f'<@{user}>, 只支持下列格式的文件 [{", ".join(self.allowed_filetypes)}]', thread_ts=thread_ts)
            return None, None

        if file["size"] > self.max_file_size:
            say(f'<@{user}>, 文件大小超过限制 ({self.max_file_size / 1024 / 1024}MB)', thread_ts=thread_ts)
            return None, None

        return self.process_file_upload(file, user, event, say)

    def process_file_upload(self, file, user, event, say):
        url_private = file["url_private"]
        filetype = file["filetype"]
        temp_file_path = self.index_cache_dir / user
        temp_file_path.mkdir(parents=True, exist_ok=True)
        temp_file_filename = temp_file_path / file["name"]
        with open(temp_file_filename, "wb") as f:
            response = requests.get(url_private, headers={"Authorization": "Bearer " + self.client.token})
            f.write(response.content)
            logging.info(f'文件已下载至 {temp_file_filename}')
            temp_file_md5 = md5(temp_file_filename)
            file_md5_name = self.index_cache_dir / (temp_file_md5 + '.' + filetype)
            if not file_md5_name.exists():
                print(f'=====> Rename file to {file_md5_name}')
                temp_file_filename.rename(file_md5_name)
                if filetype in self.voice_extension_allowed:
                    voicemessage = get_text_from_whisper(file_md5_name)
            return file_md5_name, voicemessage

    def extract_and_format_dialog(self, event, voicemessage):
        try:
            return format_dialog_text(event["text"], voicemessage)
        except Exception as e:
            logging.error(e)
            return None

    def extract_urls_from_event(self, event):
        # 从事件中提取 URL
        if 'blocks' not in event:
            return None
        urls = set()
        for block in event['blocks']:
            for element in block['elements']:
                for e in element['elements']:
                    if e['type'] == 'link':
                        url = e['url']
                        urls.add(url)
        return list(urls)

    def update_thread_history(self, thread_ts, message_str=None, urls=None, file=None):
        # 更新线程历史记录
        if thread_ts not in self.thread_message_history:
            self.thread_message_history[thread_ts] = {'dialog_texts': [], 'context_urls': set(), 'file': None}

        if urls:
            self.thread_message_history[thread_ts]['context_urls'].update(urls)

        if message_str:
            dialog_texts = self.thread_message_history[thread_ts]['dialog_texts']
            dialog_texts.append(message_str)
            if len(dialog_texts) > self.MAX_THREAD_MESSAGE_HISTORY:
                dialog_texts = dialog_texts[-self.MAX_THREAD_MESSAGE_HISTORY:]
            self.thread_message_history[thread_ts]['dialog_texts'] = dialog_texts

        if file:
            self.thread_message_history[thread_ts]['file'] = file

    def process_conversation(self, thread_ts, event, say, user, current_thread_ts, dialog_text, voicemessage):
        # 处理对话，生成回复
        urls = self.thread_message_history[thread_ts]['context_urls']
        file = self.thread_message_history[thread_ts]['file']

        # 根据对话内容，文件，URL调用LangChain或其他逻辑生成回复
        future = self.executor.submit(self.langchain_exec, dialog_text, file, urls)

        try:
            gpt_response, total_llm_model_tokens, total_embedding_model_tokens = future.result(timeout=300)
            self.update_token_usage(user, total_llm_model_tokens, total_embedding_model_tokens)
            self.update_thread_history(thread_ts, 'chatGPT: %s' % f'{gpt_response}')

            if voicemessage is None:
                say(f'<@{user}>, {gpt_response}', thread_ts=current_thread_ts)
            else:
                voice_file_path = get_voice_file_from_text(str(gpt_response))
                slack_response = self.client.files_upload_v2(file=voice_file_path, channel=event["channel"], thread_ts=current_thread_ts)
        except concurrent.futures.TimeoutError:
            future.cancel()
            err_msg = '任务超时（5分钟）并已取消。'
            say(f'<@{user}>, {err_msg}', thread_ts=current_thread_ts)

    def langchain_exec(self, dialog_text, file, urls):
        # LangChain或其他逻辑的执行部分，需要根据具体情况实现
        pass
    
    def check_usage(self, chat_id: str) -> bool:
        if not self.usage.exists(chat_id):
            self.usage.add_user(chat_id)
        if self.usage.usage_exceeded(chat_id):
            self.send_messages()
            return False
        return True
