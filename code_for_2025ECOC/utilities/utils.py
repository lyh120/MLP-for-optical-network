# import sys
# import os
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), './utilities'))
# sys.path.append(parent_dir)

import numpy as np
from scipy.special import erfcinv
import time
from scipy.stats import qmc
from .tencent import ini_TRx, ini_TencentOA, ini_OCMs, query_setting
from concurrent.futures import ThreadPoolExecutor
# from .loadfilter import *


"""
    Tx ─── mux ─── edfa1(OCM1) ─── fiber[0] ─── edfa2(OCM2) ─── fiber[1] ─── edfa3(OCM3)  ──┐
                                                                                            │
                                                                                            fiber[2]
                                                                                            │
    Rx ─── mux ─── (OCM6)edfa6 ─── fiber[4] ─── (OCM5)edfa5 ─── fiber[3] ─── (OCM4)edfa4  ──┘
"""

### Initialization from tencent.py
L1,L2,L3,L4,L5,L6=ini_TRx()
Ls = [L1, L2, L3, L4, L5]
# Ls = [L1, L4, L6]

b15, p15, iA2B, iB2A, b12, p12 = ini_TencentOA()
OA_list = [b15, iA2B, p12, b12, iB2A, p15]

OCM1, OCM2, OCM3, OCM4, OCM5, OCM6 = ini_OCMs()
OCM_list = [OCM1, OCM2, OCM3, OCM4, OCM5, OCM6]

def ber_to_q(ber):
    ber = np.asarray(ber)
    q = np.zeros(ber.shape)
    # 避免对数无穷大
    ber[ber < 1e-10] = 1e-10
    q[ber>=0.5] = 0
    q[ber<0.5] = 20*np.log10(np.sqrt(2) * erfcinv(2 * ber[ber<0.5]))
    return q

def query_ber_for_Ls(L, avg_times):
    results = []
    for _ in range(avg_times):
        _, _, avg, _ = L.query_ber()
        results.append(avg)
    return sum(results)

def query_parallel(avg_times=3):
    time.sleep(1)
    with ThreadPoolExecutor(max_workers=len(Ls)) as executor:
        futures = [executor.submit(query_ber_for_Ls, L, avg_times) for L in Ls]
        results = [future.result() for future in futures]
    ber = np.array(results) / avg_times
    return ber_to_q(ber)

def set_gain(oa, value):
    oa.set_gain(value)

def set_tilt(oa, value):
    oa.set_tilt(value)

def apply_parallel(x):
    gain, tilt = query_setting()
    real = gain + tilt
    
    with ThreadPoolExecutor(max_workers=len(OA_list)) as executor:
        # 并行设置增益
        gain_futures = [
            executor.submit(set_gain, oa, x[i])
            for i, oa in enumerate(OA_list)
            if real[i] != x[i]
        ]
        
        # 等待所有增益设置完成
        for future in gain_futures:
            future.result()
        
        # 并行设置倾斜
        tilt_futures = [
            executor.submit(set_tilt, oa, x[i + len(OA_list)])
            for i, oa in enumerate(OA_list)
            if real[i + len(OA_list)] != x[i + len(OA_list)]
        ]
        
        # 等待所有倾斜设置完成
        for future in tilt_futures:
            future.result()
        time.sleep(2)
        
# def apply(x):
#     gain, tilt = query_setting()
#     real = gain + tilt
#     loop = 0
#     for oa in OA_list:
#         if(real[loop]==x[loop]):
#             loop += 1
#             continue
#         else:
#             oa.set_gain(x[loop])
#             loop += 1
#     for oa in OA_list:
#         if(real[loop]==x[loop]):
#             loop += 1
#             continue
#         else:
#             oa.set_tilt(x[loop])
#             loop += 1

# def query_settings_sorted():
#     real_gains, real_tilts = query_setting()
#     gg = [real_gains[i] for i in [0, 2, 5, 4, 3, 1]]
#     tt = [real_tilts[i] for i in [0, 2, 5, 4, 3, 1]]
#     return gg, tt

### query input power parellel
def query_single_input_power(OA):
    _, _, Pin_avg, _ = OA.query_input_power()
    return Pin_avg
def query_all_input_power():
    with ThreadPoolExecutor() as executor:
        Pin = list(executor.map(query_single_input_power, OA_list))
    return np.array(Pin)

### query output power parellel
def query_single_output_power(OA):
    _, _, Pout_avg, _ = OA.query_output_power()
    return Pout_avg
def query_all_output_power():
    with ThreadPoolExecutor() as executor:
        Pout = list(executor.map(query_single_output_power, OA_list))
    return np.array(Pout)

### query ocm parellel
def query_single_ocm(ocm):
    psd, f = ocm.get()
    return psd, f
def query_all_ocm():
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(query_single_ocm, OCM_list))
    PSD, f = zip(*results)  # 解压结果
    return np.array(PSD), np.array(f)


def input_power(index):
    oa = OA_list[index-1]
    inst, max, avg, min = oa.query_input_power()
    return avg
def output_power(index):
    oa = OA_list[index-1]
    inst, max, avg, min = oa.query_output_power()
    return avg

def setEDFAs(gains, tilts):
    
    loop = 0
    for oa in OA_list:
        flag1 = oa.set_gain(gains[loop])
        if tilts is None:
            flag2 = True
        else:
            flag2 = oa.set_tilt(tilts[loop])

        retrynum = 0
        while not (flag1 and flag2):
            flag1 = oa.set_gain(gains[loop])
            flag2 = oa.set_tilt(tilts[loop])
            retrynum += 1
            print(f'失败，第{retrynum+1}/10次重试')
            if retrynum > 9:
                raise Exception("失败了")
            
        loop += 1
    real_gains, real_tilts = query_settings_sorted()
    return real_gains, real_tilts
    
# def setEDFA1(g, t):
#     b15.set_gain(g)
#     b15.set_tilt(t)
#     real_gain, real_tilt = query_settings_sorted()
#     return real_gain[0], real_tilt[0]
# def setEDFA2(g, t):
#     iA2B.set_gain(g)
#     iA2B.set_tilt(t)
#     real_gain, real_tilt = query_settings_sorted()
#     return real_gain[1], real_tilt[1]
# def setEDFA3(g, t):
#     p12.set_gain(g)
#     p12.set_tilt(t)
#     real_gain, real_tilt = query_settings_sorted()
#     return real_gain[2], real_tilt[2]
# def setEDFA4(g, t):
#     b12.set_gain(g)
#     b12.set_tilt(t)
#     real_gain, real_tilt = query_settings_sorted()
#     return real_gain[3], real_tilt[3]
# def setEDFA5(g, t):
#     iB2A.set_gain(g)
#     iB2A.set_tilt(t)
#     real_gain, real_tilt = query_settings_sorted()
#     return real_gain[4], real_tilt[4]
# def setEDFA6(g, t):
#     p15.set_gain(g)
#     p15.set_tilt(t)
#     real_gain, real_tilt = query_settings_sorted()
#     return real_gain[5], real_tilt[5]

def add_wavelength_batch(index):
    L_index = Ls[index-1]
    flag1 = L_index.up_tunnel()
    flag2 = dummy_signal_reshape()
    return flag1 & flag2

def drop_wavelength_batch(index):
    L_index = Ls[index-1]
    flag1 = L_index.down_tunnel()
    flag2 = dummy_signal_reshape()
    return flag1 & flag2

def dummy_signal_reshape():
    states = [L.tunnel_state() for L in Ls]
    index_of_occ = np.zeros(115)
    real_indx = [14, 19, 24, 29, 34, 39]
    for i in range(len(states)):

        if states[i]=="allocate":
            index_of_occ[real_indx[i]-1-2:real_indx[i]-1+3]=0
        elif states[i]=="implement":
            index_of_occ[real_indx[i]-1-2:real_indx[i]-1+3]=1
        else:
            print("Dummy Signal Reshape ERROR!")
            return False
    # loadfilter_waveshaper4000s_allpass()
    # channel_occ_simulate(index_of_occ)
    print("add finish")
    return True

# def check_osc_b15():
#     # 此处真正使用时，应当查询当前的log
#     # alarm = b15.query_alarm_current()
    
#     # 实验中用查询历史log的方式来展示:
#     alarm = b15.query_alarm_log()
#     return check_alarm(OA_list, alarm)

if __name__=='__main__':
    # drop_wavelength_batch(2)
    # add_wavelength_batch(3)
    # iA2B.atten(1.5)
    # print(query_settings_sorted())
    # L1.update_tunnelpower(-2)
    # print(input_power(6))
    dummy_signal_reshape()
    # print(tilt)