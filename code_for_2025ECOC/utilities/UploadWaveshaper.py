import requests
import json
import numpy as np
import pandas as pd
import copy
import time
from WSMethods import *

# ws_data = pd.read_csv('spectrum_for_3000.txt')

ip_waveshaper = 'WS201151.local'
result = requests.get('http://' + ip_waveshaper + '/waveshaper/devinfo').json()

ws_step_sum = 4000
wsFreq = np.arange(192.076, 196.076, 0.001)  # 1530-1561 nm
wsPhase = np.zeros(wsFreq.shape)
wsPort = np.ones(wsFreq.shape)

'''
def uploadWaveshaper(test_num):  # 要写入数据的次数
    ws_attn = np.zeros(ws_step_sum)
    for ws_step in range(ws_step_sum):
        channel = ws_step // 100

        if (ws_step % 100) < 50:
            ws_attn[ws_step] = ws_data.loc[test_num][channel]
        else:
            ws_attn[ws_step] = 100

    tilt = np.array([0, 0, 0, 0.22, 0.39, 0.6, 0.76, 0.83, 1.02, 1.13, 1.18, 1.39, 1.47, 1.67, 1.9, 2.09, 2.56, 2.45, 2.67, 3.08,
                 3.01, 3.1, 3.18, 3.29, 3.16, 3.58, 3.61, 3.49, 3.82, 4.68, 5.37, 6.48, 7.71, 8.68, 10.02, 11.33, 12.04, 12.14, 12.56, 12.05]) * 1.2

    for ws_step in range(ws_step_sum):
        channel2 = ws_step // 100
        ws_attn[ws_step] += tilt[channel2]

        # ws_attn[ws_step] = ws_data.loc[test_num][channel]
    wsAttn = copy.deepcopy(ws_attn)
    ws_r = uploadProfile(ip_waveshaper, wsFreq, wsAttn, wsPhase, wsPort)
    print('upload_waveshaper test_num=', test_num)
'''

def sweep_waveshaper(pass_channel_id, attn_id):
    ws_attn = np.zeros(ws_step_sum)
    for ws_step in range(ws_step_sum):
        channel = ws_step // 50
        if channel == pass_channel_id:
            ws_attn[ws_step] = attn_id*2+4
        else:
            ws_attn[ws_step] = 60

    print(ws_attn)
    wsAttn = copy.deepcopy(ws_attn)
    ws_r = uploadProfile(ip_waveshaper, wsFreq, wsAttn, wsPhase, wsPort)
    
def sweep_waveshaper_first(pass_channel_id):
    ws_attn = np.zeros(ws_step_sum)
    for ws_step in range(ws_step_sum):
        channel = ws_step // 50
        if channel == pass_channel_id:
            ws_attn[ws_step] = 0
        else:
            ws_attn[ws_step] = 60

    print(ws_attn)
    wsAttn = copy.deepcopy(ws_attn)
    ws_r = uploadProfile(ip_waveshaper, wsFreq, wsAttn, wsPhase, wsPort)

def full_load(attn_id):
    ws_attn = np.zeros(ws_step_sum) + attn_id*2+4

    print(ws_attn)
    wsAttn = copy.deepcopy(ws_attn)
    ws_r = uploadProfile(ip_waveshaper, wsFreq, wsAttn, wsPhase, wsPort)

if __name__ == "__main__":
    ws_attn = np.zeros(ws_step_sum)
    for ws_step in range(ws_step_sum):
        # channel = ws_step // 50
        # if ws_step % 50 < 12 or ws_step % 50 > 37:
        #     ws_attn[ws_step] = 100
        # else:
        #     ws_attn[ws_step] = 4
        # if (channel % 2) == 0:
        #     ws_attn[ws_step] = 0
        # else:
        #     ws_attn[ws_step] = 0

        ws_attn[ws_step] = 0  # 全通
        
        # if channel == 1:
        #     ws_attn[ws_step] = 0
        # else:
        #     ws_attn[ws_step] = 60

    # tilt = np.array([0, 0, 0, 0.22, 0.39, 0.6, 0.76, 0.83, 1.02, 1.13, 1.18, 1.39, 1.47, 1.67, 1.9, 2.09, 2.56, 2.45, 2.67, 3.08,
    #         3.01, 3.1, 3.18, 3.29, 3.16, 3.58, 3.61, 3.49, 3.82, 4.68, 5.37, 6.48, 7.71, 8.68, 10.02, 11.33, 12.04,
    #         12.14, 12.56, 12.05]) * 1.2

    # for ws_step in range(ws_step_sum):
    #     channel = ws_step // 100
    #     ws_attn[ws_step] += tilt[channel]

    print(ws_attn)
    wsAttn = copy.deepcopy(ws_attn)
    ws_r = uploadProfile(ip_waveshaper, wsFreq, wsAttn, wsPhase, wsPort)
