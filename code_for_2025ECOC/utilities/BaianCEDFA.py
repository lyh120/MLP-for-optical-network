import requests
from bs4 import BeautifulSoup
import numpy as np
import re
import warnings

warnings.filterwarnings("ignore")

#### Qiu Qizhi 2023.9.19
#### Controlling code for BA EDFAs

class BainAnCEDFA:
    def __init__(self, slot):  # eg: slot='slot3'
        self.slot = slot
        self.sign_url = 'https://192.168.1.100/login.cgi'
        self.referer = 'https://192.168.1.100/Card_'+slot+'.html'
        self.cookies = self.sign_in()
        
        
    def sign_in(self):
        para = {'USERNAME': 'admin', 'PASSWORD': '123456', 'CLICK': 'Sign in' }
        response = requests.post(self.sign_url, params= para, verify=False)
        coo = response.cookies
        return coo

    def query_real_gain(self):
        response = requests.get(url = self.referer,  cookies = self.cookies, verify=False)
        response_text = response.text  # 用实际的响应文本替换这里的'...'
        # 使用Beautiful Soup解析HTML
        soup = BeautifulSoup(response_text, 'html.parser')
        # 查找包含 "Current Gain(dB)" 的标签
        current_gain_tag = soup.find(string="Current Gain(dB):")
        if current_gain_tag:
            # 获取当前标签的下一个兄弟标签，通常包含数字
            next_data = current_gain_tag.find_next()
            if next_data:
                value = next_data.contents
                value = np.array(value, dtype='float')
                value =value.item()
                print("Real Gain of "+self.slot+"(dB):", value)
                return value
            else:
                print("未找到 Current Gain(dB) 的值")
        else:
            print("未找到 Current Gain(dB) 标签")
            
    def query_setting_gain(self):
        response = requests.get(url = self.referer,  cookies = self.cookies, verify=False)
        response_text = response.text  # 用实际的响应文本替换这里的'...'
        # 使用Beautiful Soup解析HTML
        soup = BeautifulSoup(response_text, 'html.parser')
        # 查找包含 "Current Gain(dB)" 的标签
        current_gain_tag = soup.find(string="AGC Gain Setting(13.0 ~ 25.0 dB):")
        if current_gain_tag:
            # 获取当前标签的下一个兄弟标签，通常包含数字
            next_data = current_gain_tag.find_next()
            if next_data:
                value = next_data.contents
                value = np.array(value, dtype='float')
                value =value.item()
                print("Setting Gain of "+self.slot+"(dB):", value)
                return value
            else:
                print("未找到 AGC Gain Setting 的值")
        else:
            print("未找到 AGC Gain Setting 标签")
            
    def query_tilt(self):
        response = requests.get(url = self.referer,  cookies = self.cookies, verify=False)
        response_text = response.text  # 用实际的响应文本替换这里的'...'
        # 使用Beautiful Soup解析HTML
        soup = BeautifulSoup(response_text, 'html.parser')
        # 查找包含 "Current Gain(dB)" 的标签
        current_gain_tag = soup.find(string="Gain Tilt Setting(-2.0 ~ 2.0 dB):")
        if current_gain_tag:
            # 获取当前标签的下一个兄弟标签，通常包含数字
            next_data = current_gain_tag.find_next()
            if next_data:
                value = next_data.contents
                value = np.array(value, dtype='float')
                value =value.item()
                print("Gain Tilt Setting: "+self.slot+"(dB):", value)
                return value
            else:
                print("未找到 Gain Tilt Setting的值")
        else:
            print("未找到Gain Tilt Setting标签")


    def set_gain(self, gain): #13~25dB
        gain = round(gain,1)
        if gain<13 or gain>25:
            print(self.slot + " Gain超过可设置范围13~25dB!")
            return 0
        
        headers = {
            'Origin': 'https://192.168.1.100',
            'Referer': self.referer
        }

        data = {
            'configtxt1': str(gain),
            'setcurrent': 'Set'
        }

        response = requests.post('https://192.168.1.100/edfaconfout.cgi', cookies=self.cookies, headers=headers, data=data, verify=False)
        response = response.text
        # 使用正则表达式匹配文本内容
        match = re.search(r'alert\("([^"]+)"\);', response)
        if match:
            extracted_text = match.group(1)
            print(extracted_text)
        else:
            print(self.slot + "设置gain报错")

    def set_tilt(self, tilt):
        tilt = round(tilt,1)
        if tilt<-2 or tilt>2:
            print(self.slot + " Tilt超过可设置范围-2~2!")
            return 0
        headers = {
            'Origin': 'https://192.168.1.100',
            'Referer': self.referer
        }

        data = {
            'configgaintilt1': str(tilt),
            'setagaintilt': 'Set'
        }

        response = requests.post('https://192.168.1.100/edfagaintiltsetting.cgi', cookies=self.cookies, headers=headers, data=data, verify=False)
        response = response.text
        # 使用正则表达式匹配文本内容
        match = re.search(r'alert\("([^"]+)"\);', response)
        if match:
            extracted_text = match.group(1)
            print(extracted_text)
        else:
            print(self.slot + "设置tilt报错")
            
            
def ini_BainanC_OA():
    slot2 = BainAnCEDFA(slot='slot2')
    slot3 = BainAnCEDFA(slot='slot3')
    slot6 = BainAnCEDFA(slot='slot6')
    return slot2, slot3, slot6


# url = 'https://192.168.1.47/login.cgi'
# para = {'USERNAME': 'admin', 'PASSWORD': 'Admin@123', 'CLICK': 'Sign in' }
# response = requests.post(url, params= para, verify=False)

# res = response.text
# coo = response.cookies
# print(res)
# print(coo)


# cookies = {
#     'cookiename': 'admin',
#     'session_id': 'bGFuZ3VhZ2U9MSZ1c2VyX25hbWU9YWRtaW4maXA9MTkyLjE2OC4xLjUxJmxvZ2luX3RpbWU9MzUxNzU0NQ==',
# }
# headers = {
#     'Origin': 'https://192.168.1.47',
#     'Referer': 'https://192.168.1.47/Card_slot3.html'
# }

# data = {
#     'configtxt1': '16.2',
#     'setcurrent': 'Set'
# }

# response = requests.post('https://192.168.1.47/edfaconfout.cgi', cookies = coo, headers=headers, data=data, verify=False)
# print(response.text)

# response = requests.get('https://192.168.1.47/Card_slot3.html',  cookies = coo, verify=False)


# response_text = response.text  # 用实际的响应文本替换这里的'...'
# # 使用Beautiful Soup解析HTML
# soup = BeautifulSoup(response_text, 'html.parser')

# # 查找包含 "Current Gain(dB)" 的标签
# current_gain_tag = soup.find(text="Current Gain(dB):")

# if current_gain_tag:
#     # 获取当前标签的下一个兄弟标签，通常包含数字
#     next_data = current_gain_tag.find_next()
#     if next_data:
#         value = next_data.contents
#         value = np.array(value, dtype='float')
#         print("Current Gain(dB):", value)
#     else:
#         print("未找到 Current Gain(dB) 的值")
# else:
#     print("未找到 Current Gain(dB) 标签")
