from flask import Flask, jsonify
from flask_apscheduler import APScheduler
from dotenv import load_dotenv
from slack_bolt.adapter.flask import SlackRequestHandler

from slack.slack_api import init_slack_app
from slack.article_push import schedule_articles


# 加载环境变量
load_dotenv()


def main():
    app = Flask(__name__)

    slack_app = init_slack_app()

    scheduler = APScheduler()
    scheduler.api_enabled = True
    scheduler.init_app(app)
    scheduler.add_job(
        func=lambda: schedule_articles(slack_app.client),
        trigger='cron',
        id='daily_news_task',
        hour=1,
        minute=30
    )
    scheduler.start()

    slack_handler = SlackRequestHandler(slack_app)

    @app.route("/webhook/events", methods=["POST"])
    def slack_events():
        return slack_handler.handle(request)

    @app.route('/ping')
    def ping():
        return jsonify({'message': 'pong'})

    app.run(debug=True)

if __name__ == '__main__':
    main()
