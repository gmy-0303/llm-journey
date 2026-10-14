import os
import logging
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential 
# retry（重试）、stop_after_attempt（尝试几次后停止）、
# wait_exponential（等待时间按指数增加）。这些是用来处理网络卡顿的。
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
"""日志是有“级别”的级：
DEBUG（调试）：最细枝末节的废话（比如“变量x现在的值是5”）。
INFO（信息）：正常的重要操作。
WARNING（警告）：有点小问题，但还能跑。
ERROR（错误）：出问题了
CRITICAL（严重）：系统崩了 """

"""level=logging.INFO 的意思是：DEBUG不记，从 INFO 开始记。
format='...'：“每条记录按我规定的格式来写。”
%(asctime)s：时间。日志产生的那一刻。
%(levelname)s：级别名称。比如“INFO”、“ERROR”。
%(message)s：具体内容。你写在 logging.info("...") 括号里的那句话。
中间的 - 和空格：装饰，让它们隔开好看一点。"""

"""日志里就会打印出类似这样的话：
2026-10-14 15:30:00,123 - INFO - 请求 API 中...
2026-10-14 15:30:02,456 - ERROR - 网络连接失败
时间 - 级别 - 内容。"""

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)
# client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")：创建 OpenAI 客户端，指定 API 密钥和 API 地址。
# os.environ["DEEPSEEK_API_KEY"]：从环境变量中获取 API 密钥。这个密钥是 DeepSeek 提供的，用于访问他们的 API。   
# $env:DEEPSEEK_API_KEY = "api"
# python .\chat.py
# @retry(stop=stop_after_attempt(3),wait=wait_exponential(multiplier=1, min=2, max=10))
# retry 装饰器的意思是：如果函数执行失败，就重试。
# stop_after_attempt(3) 的意思是：最多尝试 3 次。
# wait_exponential(multiplier=1, min=2, max=10) 的意思是：每次重试的等待时间按指数增加，最少 2 秒，最多 10 秒。
def ask_llm(user_input):
    # ask_llm 函数的作用是：向 LLM（大语言模型）提问，并返回回答。
    logging.info(f"请求 API 中...")  
    # logging.info(f"请求 API 中...")：记录日志，表示正在向 LLM 提问。
    response = client.chat.completions.create(
        # client.chat.completions.create：调用 OpenAI 的 chat.completions.create 方法。
        # model="deepseek-chat"：指定使用的模型。这里使用的是 DeepSeek 的 Chat 模型。   
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": user_input}
            # messages：消息列表。这里只有一条消息，是用户输入的。
            # {"role": "user", "content": user_input}：表示用户输入的内容。
        ]
    )
    return response.choices[0].message.content
# 返回 LLM 的回答。response 是 API 返回的结果，choices 是可能的回答列表，取第一个回答，再取它的 message，再取它的 content，就是最终的回答内容。
# .message.content：拿里面真正的文字内容。
if __name__ == "__main__":
    try:
        user_input = input("请输入你的问题：")
        # user_input = input("请输入你的问题：")：提示用户输入问题。
        response = ask_llm(user_input)
        # response = ask_llm(user_input)：调用 ask_llm 函数，向 LLM 提问，并获取回答。
        print(f"LLM 的回答是：{response}")
    except Exception as e:
        logging.error(f"发生错误：{e}")
