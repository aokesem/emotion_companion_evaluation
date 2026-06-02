from openai import OpenAI

# 初始化客户端（从环境变量读取 OPENAI_API_KEY）
client = OpenAI()

# Chat Completions 接口
completion = client.chat.completions.create(
    model="gpt-5.2",
    messages=[
        {"role": "user", "content": "Hello, GPT"},
    ],
    max_tokens=1024,
)
print(completion.choices[0].message.content)

# Responses 接口
response = client.responses.create(
    model="gpt-5.2",
    input="Hello, GPT",
)
print(response.output_text)