import requests

# 定义 FastAPI 服务的 URL
url = "http://127.0.0.1:8102/infer_bertvits2/"

# 定义请求体内容
request_data = {
    "speaker_name": "八重神子_ZH",
    "language": "ZH",
    "length_scale": 1.2,
    "infer_text": "即使引导已经破碎，也请觐见艾尔登法环",
    "infer_id": 4,
    "sdp_ratio": 0.4,
    
}
## 这里用 infer_id 这个变量控制输出语音的路径。 id是index的意思。而不是identity。 
a  = request_data["infer_id"]
request_data["output_path"] =  f"A4_model_output/SSB0005_{a}.wav"

# 发送 POST 请求
response = requests.post(url, json=request_data)

# 检查响应状态并处理
if response.status_code == 200:
    # 输出服务器返回的内容
    print("请求成功：")
    print(response.json())
else:
    print(f"请求失败，状态码: {response.status_code}")
    print(response.text)