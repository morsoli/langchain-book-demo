import os
from flask import Flask, jsonify, request
from flask_apscheduler import APScheduler
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App, BoltResponse
from slack_bolt.error import BoltUnhandledRequestError

from slack.slack_api import SlackAPIHandler
from slack.article_push import schedule_articles
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

app = Flask(__name__)  # 初始化一个Flask应用实例

# 初始化Slack应用
slack_app = App(
    token=os.environ.get("SLACK_TOKEN"),  # 从环境变量获取Slack的OAuth访问令牌
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),  # 从环境变量获取Slack的签名密钥
    raise_error_for_unhandled_request=True  # 对于未处理的请求抛出错误
)

slack_handler = SlackRequestHandler(slack_app)  # 创建一个Slack请求处理器
slack_api_handler = SlackAPIHandler(slack_app)  # 创建一个处理Slack API事件的处理器

@app.route("/webhook/events", methods=["POST"])  # 定义一个路由来处理来自Slack的事件
def slack_events():
    return slack_handler.handle(request)  # 使用Slack请求处理器来处理请求

# 定义错误处理器
@slack_app.error
def handle_errors(error):
    if isinstance(error, BoltUnhandledRequestError):  # 未处理的请求错误
        return BoltResponse(status=200, body="")
    else:
        return BoltResponse(status=500, body="出错了！")  # 其他错误
    
@slack_app.event(event="message")
def handle_message(event, say, logger):
    slack_api_handler.process_event(event, say, logger)  # 处理收到的消息事件
    
# @slack_app.event("app_mention")
# def handle_mentions(event, say, logger):
#     slack_api_handler.process_event(event, say, logger)  # 处理收到的消息事件

@app.route('/ping')
def ping():
    return jsonify({'message': 'pong'})

# scheduler = APScheduler()
# scheduler.api_enabled = True
# scheduler.init_app(app)
# scheduler.add_job(
#     func=lambda: schedule_articles(slack_app.client),
#     trigger='cron',
#     id='daily_news_task',
#     hour=1,
#     minute=30
# )
# scheduler.start()
    

if __name__ == '__main__':
    app.run(debug=True)
