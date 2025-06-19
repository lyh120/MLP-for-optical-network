#import WaveShaper Python API
from wsapi import *
from utilities.WSMethods import createWspString
import numpy as np
import os
from matplotlib import pyplot as plt
import scipy

from utilities.GlobalControl import GlobalControl
try:
    logger = GlobalControl.logger
except:
    logger = GlobalControl.init_logger()
    logger.warning('GlobalControl中logger未定义！')
logger.info('loadfilter.py imported.')


CURRENT_PATH = os.getcwd()


def get_wsptext_C100L100_allpass(port_num=1):
    wsFreq = np.arange(187.6, 196.0, 0.2)
    wsPhase = np.zeros(wsFreq.shape)
    wsPort = np.ones(wsFreq.shape) * port_num
    wsAttn = np.ones(wsFreq.shape) * 2  # 基础2db衰减
    wsptext = createWspString(wsFreq, wsAttn, wsPhase, wsPort)
    
    # plt.plot(wsAttn)
    # plt.show()
    
    return wsptext


def get_wsptext_ase23dbm_gff(port_num=1):
    CURRENT_PATH = os.getcwd()
    save_path = CURRENT_PATH + '/output/'
    dict = scipy.io.loadmat(save_path + '/ASEsource_23dbm_gff.mat')
    wsFreq = dict['f'].ravel()
    wsFreq = np.round(wsFreq, 3) - 0.03
    wsPhase = np.zeros(wsFreq.shape)
    wsPort = np.ones(wsFreq.shape) * port_num
    wsAttn = dict['gff'].ravel() + 2
    wsptext = createWspString(wsFreq, wsAttn, wsPhase, wsPort)
    
    plt.plot(wsFreq, wsAttn)
    plt.show()
    
    return wsptext


def get_wsptext_C100L100(port_num=1):
    wsFreq = np.arange(187.5, 196.2, 0.050)
    wsPhase = np.zeros(wsFreq.shape)
    wsPort = np.ones(wsFreq.shape) * port_num
    wsAttn = np.random.rand(wsFreq.shape[0]) * 3
    wsptext = createWspString(wsFreq, wsAttn, wsPhase, wsPort)
    
    plt.plot(wsAttn)
    plt.show()
    
    return wsptext

def generate_random_wsptext(times=500, attn_bias=0, port_num=1):
    wsFreq = np.arange(187.5, 196.2, 0.050)
    wsPhase = np.zeros(wsFreq.shape)
    wsPort = np.ones(wsFreq.shape) * port_num
    for i in range(times):
        wsAttn = np.random.rand(wsFreq.shape[0]) * 3 + attn_bias
        wsptext = createWspString(wsFreq, wsAttn, wsPhase, wsPort)
        f = open(CURRENT_PATH + f'/random_ripple_wsp/randripple_bias_{attn_bias}_{str(i)}.wsp', "w")
        f.write(wsptext)
        f.close()


def generate_random_wsptext_w_random_bias(times=1600, port_num=1):
    wsFreq = np.arange(187.5, 196.2, 0.050)
    wsPhase = np.zeros(wsFreq.shape)
    wsPort = np.ones(wsFreq.shape) * port_num
    for i in range(times):
        attn_bias = np.random.rand(wsFreq.shape[0]) * 9
        wsAttn = np.random.rand(wsFreq.shape[0]) * 3 + attn_bias
        wsptext = createWspString(wsFreq, wsAttn, wsPhase, wsPort)
        f = open(CURRENT_PATH + f'/random_ripple_bias_wsp/rand_ripple_and_bias_{str(i)}.wsp', "w")
        f.write(wsptext)
        f.close()


def get_wsptest_C100L100_fromgenerated(attn_bias, wsp_ind):
    with open(f'./random_ripple_wsp/randripple_bias_{attn_bias}_{wsp_ind}.wsp', mode="r", encoding="utf-8") as f:
        data= f.read()   #read() 一次性读全部内容，以字符串的形式返回结果。
    f.close()
    return data


def get_wsptest_C100L100_fromgenerated_randbias(wsp_ind):
    with open(f'./random_ripple_bias_wsp/rand_ripple_and_bias_{wsp_ind}.wsp', mode="r", encoding="utf-8") as f:
        data= f.read()   #read() 一次性读全部内容，以字符串的形式返回结果。
    f.close()
    return data


def sweep_waveshaper(pass_channel_id):
    wsFreq = np.arange(187.6, 196.0, 0.2)
    wsPhase = np.zeros(wsFreq.shape)
    wsPort = np.ones(wsFreq.shape) * 1  # port 1
    wsAttn = np.ones(wsFreq.shape) * 2  # 基础2db衰减
    
    for channel in range(len(wsFreq)):
        if channel in pass_channel_id:
           pass
        else:
            wsAttn[channel] = 60

    wsptext = createWspString(wsFreq, wsAttn, wsPhase, wsPort)

    ws1 = 'ws1'
    #Create WaveShaper instance SN93_4000S and name it "ws1"
    rc = ws_create_waveshaper(ws1, "./testdata/SN200037.wsconfig")
    # logger.debug("ws_create_waveshaper, result:"+str(ws_get_result_description(rc)))

    #Open WaveShaper
    rc = ws_open_waveshaper(ws1)
    logger.debug("ws_open_waveshaper, result:"+str(ws_get_result_description(rc)))

    #Compute and load the filter to device
    rc = ws_load_profile(ws1, wsptext)
    logger.debug("ws_load_profile, result:"+str(ws_get_result_description(rc)))
    
    rc = ws_delete_waveshaper(ws1)
    # logger.debug("ws_delete_waveshaper, result:"+str(ws_get_result_description(rc)))


def loadfilter_waveshaper4000s_from_generated_wsp(attn_bias, wsp_ind):
    
    ws1 = 'ws1'
    #Create WaveShaper instance SN93_4000S and name it "ws1"
    rc = ws_create_waveshaper(ws1, "./testdata/SN200037.wsconfig")
    # logger.debug("ws_create_waveshaper, result:"+str(ws_get_result_description(rc)))

    #Open WaveShaper
    rc = ws_open_waveshaper(ws1)
    logger.debug("ws_open_waveshaper, result:"+str(ws_get_result_description(rc)))

    #read WSP from file
    # wsptext = get_wsptest_C100L100_fromgenerated(attn_bias, wsp_ind)
    wsptext = get_wsptest_C100L100_fromgenerated_randbias(wsp_ind)
    logger.info('WS load: baseline数据!')
    

    #Compute and load the filter to device
    rc = ws_load_profile(ws1, wsptext)
    logger.debug("ws_load_profile, result:"+str(ws_get_result_description(rc)))
    
    rc = ws_delete_waveshaper(ws1)
    # logger.debug("ws_delete_waveshaper, result:"+str(ws_get_result_description(rc)))


def loadfilter_waveshaper4000s_allpass():
    
    ws1 = 'ws1'
    #Create WaveShaper instance SN93_4000S and name it "ws1"
    rc = ws_create_waveshaper(ws1, "./testdata/SN200037.wsconfig")
    # logger.debug("ws_create_waveshaper, result:"+str(ws_get_result_description(rc)))

    #Open WaveShaper
    rc = ws_open_waveshaper(ws1)
    logger.debug("ws_open_waveshaper, result:"+str(ws_get_result_description(rc)))

    #read WSP from file
    # wsptext = get_wsptest_C100L100_fromgenerated(attn_bias, wsp_ind)
    wsptext = get_wsptext_C100L100_allpass()
    logger.info('WS load: 全通数据')
    

    #Compute and load the filter to device
    rc = ws_load_profile(ws1, wsptext)
    logger.debug("ws_load_profile, result:"+str(ws_get_result_description(rc)))
    
    rc = ws_delete_waveshaper(ws1)
    # logger.debug("ws_delete_waveshaper, result:"+str(ws_get_result_description(rc)))
    

def loadfilter_waveshaper4000s_ase23dbmgff():
    
    ws1 = 'ws1'
    #Create WaveShaper instance SN93_4000S and name it "ws1"
    rc = ws_create_waveshaper(ws1, "./testdata/SN200037.wsconfig")
    # logger.debug("ws_create_waveshaper, result:"+str(ws_get_result_description(rc)))

    #Open WaveShaper
    rc = ws_open_waveshaper(ws1)
    logger.debug("ws_open_waveshaper, result:"+str(ws_get_result_description(rc)))

    #read WSP from file
    # wsptext = get_wsptest_C100L100_fromgenerated(attn_bias, wsp_ind)
    wsptext = get_wsptext_ase23dbm_gff()
    logger.info('WS load: ASE Source 23dBm GFF')
    

    #Compute and load the filter to device
    rc = ws_load_profile(ws1, wsptext)
    logger.debug("ws_load_profile, result:"+str(ws_get_result_description(rc)))
    
    rc = ws_delete_waveshaper(ws1)
    # logger.debug("ws_delete_waveshaper, result:"+str(ws_get_result_description(rc)))

def channel_occ_simulate(index_of_occ = np.ones(115)):
    index_of_occ[0:3] = 0
    index_of_occ[112:] = 0
    
    real_indx = [14,
                 19,
                 24,
                 29,
                 34,
                 39]
    index_of_occ[np.array(real_indx)-1] = 0
    
    # wsFreq = np.arange(192.076, 196.076, 0.001)  # C band: 1530-1561 nm
    wsFreq = np.arange(187.276, 196.076, 0.001)  # C+L band
    wsAttn = np.ones(wsFreq.shape)*60
    wsPhase = np.zeros(wsFreq.shape)
    wsPort = np.ones(wsFreq.shape)
    
    wsAttn_for_flat = np.zeros(wsFreq.shape)
    
    # 1-3 信道 54-64信道不可用
    for index_channel in range(3,112):
        if index_of_occ[index_channel] !=0 :
            channel_indx = index_channel+1
            # channel_indx = 53
            center_fre = 196.0875 - (channel_indx-1) * 0.075
            wsFreq_channel_occ = np.arange(center_fre-0.075/2, center_fre+0.075/2, 0.001)
            wsFreq_channel_sig_occ = np.arange(center_fre-0.06391/2, center_fre+0.06391/2, 0.001)
            wsFreq_channel_sig_indx =(np.floor((wsFreq_channel_sig_occ-187.276)/0.001)).astype(int)
            wsAttn[wsFreq_channel_sig_indx] = wsAttn_for_flat[wsFreq_channel_sig_indx]

    # Upload profile
    wsptext = createWspString(wsFreq, wsAttn, wsPhase, wsPort)

    waveshaper_name = 'ws1'
    #Create WaveShaper instance SN93_4000S and name it "ws1"
    rc = ws_create_waveshaper(waveshaper_name, "./testdata/SN200037.wsconfig")
    # logger.debug("ws_create_waveshaper, result:"+str(ws_get_result_description(rc)))

    #Open WaveShaper
    rc = ws_open_waveshaper(waveshaper_name)
    logger.debug("ws_open_waveshaper, result:"+str(ws_get_result_description(rc)))

    #Compute and load the filter to device
    rc = ws_load_profile(waveshaper_name, wsptext)
    logger.debug("ws_load_profile, result:"+str(ws_get_result_description(rc)))
    
    rc = ws_delete_waveshaper(waveshaper_name)
    # logger.debug("ws_delete_waveshaper, result:"+str(ws_get_result_description(rc)))
    
    return wsAttn





if __name__ == '__main__':
    
    # for attn_bias in range(9):
    #     generate_random_wsptext(times=150, attn_bias=attn_bias, port_num=1)
    # generate_random_wsptext(times=600, attn_bias=9, port_num=1)
    # generate_random_wsptext_w_random_bias(times=1200, port_num=1)
    
    loadfilter_waveshaper4000s_allpass()
    # loadfilter_waveshaper4000s_ase23dbmgff()
    # loadfilter_waveshaper4000s_from_generated_wsp(attn_bias=0, wsp_ind=10)
    # loadfilter_waveshaper4000s_from_generated_wsp(attn_bias=0, wsp_ind=1100)

    # %%
    # ws1 = 'ws1'
    # #Create WaveShaper instance SN93_4000S and name it "ws1"
    # rc = ws_create_waveshaper(ws1, "./testdata/SN200037.wsconfig")
    # print("ws_create_waveshaper, result:"+str(ws_get_result_description(rc)))

    # #Open WaveShaper
    # rc = ws_open_waveshaper(ws1)
    # print("ws_open_waveshaper, result:"+str(ws_get_result_description(rc)))

    # #read WSP from file
    # # wspfile = open('./testdata/Bandpass_filter_fc194THz_BW100GHz_attn0dB_pt1.wsp', 'r')
    # # wsptext = wspfile.read()

    # #generate WSP
    # # wsptext = get_wsptext_C100L100()
    # wsptext = get_wsptest_C100L100_fromgenerated(12)

    # #Compute and load the filter to device
    # rc = ws_load_profile(ws1, wsptext)
    # print("ws_load_profile, result:"+str(ws_get_result_description(rc)))

