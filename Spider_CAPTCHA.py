import requests
import time
import base64
import hashlib

def get_captcha():
    url = "https://captcha.ruijie.com.cn/captcha/get"
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "Connection": "keep-alive",
        "Content-Length": "103",
        "Content-Type": "application/json; charset=UTF-8",
        "Host": "captcha.ruijie.com.cn",
        "Origin": "https://captcha.ruijie.com.cn",
        "Referer": "https://captcha.ruijie.com.cn/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36 Edg/134.0.0.0",
        "sec-ch-ua": "\"Chromium\";v=\"134\", \"Not:A-Brand\";v=\"24\", \"Microsoft Edge\";v=\"134\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
    }
    date = { "captchaType": "clickWord", "clientUid": "point-fda3985b-6805-45ff-8f2f-3c15aa77e97a", "ts": int(time.time() * 1000) }

    response = requests.post(url, headers=headers, json=date)

    # base64转图片
    img_data = base64.b64decode(response.json()['repData']['originalImageBase64'])
    # 文件名为图片md5 + .png  保存到 Img_sets目录下
    img_name = hashlib.md5(img_data).hexdigest() + '.png'
    with open(f'Img_sets/{img_name}', 'wb') as f:
        f.write(img_data)




if __name__ == '__main__':
    # 获取200张
    for i in range(200):
        get_captcha()
        time.sleep(2)
