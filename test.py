import requests

url = "https://api.moonshot.cn/v1/models"

headers = {"Authorization": "Bearer sk-q2bK7LDXf1fkgDwOnmckvjkBeuU8vbDpLEZ54ShIg0hXlWQo"}

response = requests.get(url, headers=headers)

print(response.text)