prompt_data = {
    "gpt-novelist": {
        "name": {
            "cn": "小说家"
        },
        "prompt": {
            "cn": "写一本拥有出人意料结局的推理小说。\n编写一个有关科技创新的未来世界的小说。\n创造一个让读者感到沉浸其中的幻想故事。"
        }
    },
}

def register_slack_slash_commands(slack_app):
    slack_app.command("/gpt-as-novelist")(handle_command_gpt_as_novelist)

def get_command_name(command):
    return command["command"].replace("/", "")

def build_prompt_blocks(prompt_key):
    return [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{prompt_data[prompt_key]['name']['cn']}"
            }
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"{prompt_data[prompt_key]['prompt']['cn']}"
                },
            ]
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": "---\n请复制上述提示词之一向我提问"
                }
            ]
        },
    ]

def handle_command_gpt_as_novelist(ack, say, command):
    ack()
    channel_id = command["channel_id"]
    user_id = command["user_id"]
    blocks = build_prompt_blocks(get_command_name(command))

    say(channel=channel_id,
        text=f"<@{user_id}>, 开始吧!",
        blocks=blocks,
        reply_broadcast=True
    )