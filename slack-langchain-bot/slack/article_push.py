from datetime import date
import feedparser
from typing import List, Dict, Optional

TODAY = date.today()
MAX_POSTS = 10

def get_summary_from_gpt(url: str) -> str:
    """从给定的 URL 获取博客摘要（此处为伪实现）。

    Args:
        url (str): 博客文章的 URL。

    Returns:
        str: 博客摘要。
    """
    blog_summary_prompt = '请用中文简短概括这篇文章的内容。'
    return blog_summary_prompt

def parse_feed_entry(entry) -> Dict[str, Optional[str]]:
    """解析 RSS 源中的一条记录。

    Args:
        entry: RSS 源中的一条记录。

    Returns:
        Dict[str, Optional[str]]: 包含博客标题、摘要、URL和发布日期的字典。
    """
    try:
        gpt_answer = get_summary_from_gpt(entry.link)
    except Exception as e:
        print(e)
        gpt_answer = None
    
    published_time = entry.published_parsed if 'published_parsed' in entry else None
    return {
        'title': entry.title,
        'summary': gpt_answer,
        'url': entry.link,
        'publish_date': published_time
    }

def get_awesome_article(rss_url: str) -> List[Dict[str, Optional[str]]]:
    """从 RSS 源获取最新的博客列表。

    Args:
        rss_url (str): RSS 源的 URL。

    Returns:
        List[Dict[str, Optional[str]]]: 包含博客信息的列表。
    """
    feed = feedparser.parse(rss_url)
    return [parse_feed_entry(entry) for entry in feed.entries[:MAX_POSTS]]

def build_slack_blocks(title: str, blog: List[Dict[str, Optional[str]]]) -> List[Dict]:
    """构建用于 Slack 显示的博客块。

    Args:
        title (str): 消息标题。
        blog (List[Dict[str, Optional[str]]]): 包含博客信息的列表。

    Returns:
        List[Dict]: Slack 消息块。
    """
    blocks = [{
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f"{title} # {TODAY.strftime('%Y-%m-%d')}"
        }
    }]
    
    for blog_item in blog:
        blocks.extend([
            {
                "type": "section",
                "text": {
                    "text": f"*{blog_item['title']}*",
                    "type": "mrkdwn"
                },
            },
            {
                "type": "section",
                "text": {
                    "text": f"{blog_item['summary']}",
                    "type": "plain_text"
                },
            },
            {
                "type": "section",
                "text": {
                    "text": f"原文链接：<{blog_item['url']}>",
                    "type": "mrkdwn"
                },
            },
            {"type": "divider"}
        ])
    
    return blocks

def build_awesome_article() -> List[Dict]:
    """构建热门博客块。

    Args:
        blog_key (str): 博客关键词或分类。

    Returns:
        List[Dict]: 用于 Slack 显示的热门博客块。
    """
    awesome_article = get_awesome_article("https://liduos.com/atom.xml")
    return build_slack_blocks("博客", awesome_article)

def schedule_articles(client):
    """关注博客文章定时推送"""
    article = build_awesome_article()
    try:
        response = client.chat_postMessage(
            channel="每日文章精选",
            text="观点",
            blocks=article,
            reply_broadcast=True,
            unfurl_links=False,
        )
        print(response)
    except Exception as e:
        print(e)