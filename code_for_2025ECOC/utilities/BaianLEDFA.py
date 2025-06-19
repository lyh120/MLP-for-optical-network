import requests
import numpy as np
import re
import warnings
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import os
warnings.filterwarnings("ignore")

#### Qiu Qizhi 2023.9.19
#### Controlling code for BA EDFAs

class BainAnLEDFA:
    def __init__(self, ip):
        self.ip = ip
        self.sign_url = 'http://192.168.1.'+str(ip)+'/login.cgi'
        self.referer = 'http://192.168.1.'+str(ip)+'/io_cgi.ssi'
        self.conf = 'http://192.168.1.'+str(ip)+'/config1.cgi'
        self.query_url = 'http://192.168.1.'+str(ip)+'/io_http.ssi'
        # self.cookies = self.sign_in()
        
    # def sign_in(self):
    #     para = {'USERNAME': 'admin', 'PASSWORD': '123456', 'CLICK': 'Sign+in'}
    #     response = requests.post(self.sign_url, params= para, verify=False)
    #     coo = response.cookies
    #     return coo

    def query_setting_gain(self):
        response = requests.get(url = self.referer, verify=False)
        response_text = response.text
        # 定义正则表达式模式
        pattern = r'OUT Gain Goal:</td><td><!--#outatt-->(\d+\.\d+)\s*dB'
        # 在response中搜索匹配的内容
        match = re.search(pattern, response_text)
        # 如果找到匹配的内容，则提取数值部分
        if match:
            value = float(match.group(1))
            value = round(value,1)
            print("L-EDFA(ip:"+str(self.ip)+") OUT Gain Goal:", value, "dB")
            return value
        else:
            print("未找到匹配的内容")

    def query_real_gain(self):
        response = requests.get(url = self.query_url, verify=False)
        response_text = response.text
        
        # 提取 Input Power
        input_power_match = re.search(r'Input Power:</td><td><!--#inpower-->(.*?)</td>', response_text)
        input_power = input_power_match.group(1).strip() if input_power_match else None
        if input_power == None:
            print("查询L-EDFA(ip:"+str(self.ip)+") 实际增益时报错")
            return -1
        elif input_power=='Low':
            print("Inout Power of L-EDFA(ip:"+str(self.ip)+") is Low")
            return -1
        else:
            input_power = float(input_power[:-4])
            input_power = round(input_power,1)
        # 提取 Output Power
        output_power_match = re.search(r'Output Power:</td><td><!--#outpower-->(.*?)</td>', response_text)
        output_power = output_power_match.group(1).strip() if output_power_match else None
        output_power = float(output_power[:-4])
        
        real_gain = round((output_power - input_power), 1)
        print("L-EDFA(ip:"+str(self.ip)+") OUT Gain Real:", real_gain, "dB")
        return real_gain

    def set_gain(self, gain): #7.0~17 dB 利用selenium
        gain = round(gain,1)
        if (gain > 17) or (gain < 7):
            print("L-EDFA增益范围为7~17！请重新设置")
            return 0
        chrome_options = Options()
        chrome_options.add_argument('--headless') #无头模式
        driver = webdriver.Chrome(options=chrome_options)
        # 打开网页
        driver.get(self.referer)
        # 找到需要填写数据的元素，并输入数据
        opoout_input = driver.find_element(By.NAME, 'OPOUT')
        opoout_input.send_keys(str(gain))
        # 提交表单
        submit_button = driver.find_element(By.NAME,'Display')
        submit_button.click()
        # os.system('taskkill /im chromedriver.exe /F')
        driver.quit()
        
def ini_L_OA():
    LOA1 = BainAnLEDFA(198)
    LOA2 = BainAnLEDFA(149)
    LOA3 = BainAnLEDFA(146)
    LOA4 = BainAnLEDFA(142)
    LOA5 = BainAnLEDFA(247)
    return LOA1, LOA2, LOA3, LOA4, LOA5

