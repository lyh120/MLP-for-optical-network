from osa import *
from UploadWaveshaper import *
import numpy
from scipy import interpolate
import os
from tencent import *
def set_no_atten():
    CURRENT_PATH = os.getcwd()
    ip_waveshaper = 'WS201151.local'
    result = requests.get('http://' + ip_waveshaper + '/waveshaper/devinfo').json()
    wsFreq = np.arange(192.076, 196.076, 0.001)  # 1530-1561 nm
    wsPhase = np.zeros(wsFreq.shape)
    wsPort = np.ones(wsFreq.shape)
    wsAttn = np.zeros(wsFreq.shape)
    ws_r = uploadProfile(ip_waveshaper, wsFreq, wsAttn, wsPhase, wsPort)
    
def set_flat_spectrum(high_fre = 195.9375,low_fre = 192.1500):
    set_no_atten()
    CURRENT_PATH = os.getcwd()
    save_path = CURRENT_PATH + '/outputosatemp/'
    osa_obj_name = 'test01'
    wsFreq = np.arange(192.076, 196.076, 0.001)
    wsAttn = np.ones(wsFreq.shape)*80
    # for iterind in range(5):
    osa_obj = OSA([192.1, 196.0], osa_obj_name, save_path)
    psd, f= osa_obj.OSA_data_save(plot=0, trace_str="TRG")

    min_psd = np.min(psd)
    attenlist = psd-np.linspace(min_psd,min_psd,np.size(psd))
    wsFreq_att_modify_list = np.where ((wsFreq>=low_fre)&(wsFreq<=high_fre))
    wsFreq_att_modify  = wsFreq[wsFreq_att_modify_list]                                  
    func1=interpolate.interp1d(f,attenlist,kind='quadratic')
    wsAttn[wsFreq_att_modify_list]=func1(wsFreq_att_modify)
    wsAttn_modify = np.copy(wsAttn)
    wsPhase = np.zeros(wsFreq.shape)
    wsPort = np.ones(wsFreq.shape)
    ws_r = uploadProfile(ip_waveshaper, wsFreq, wsAttn, wsPhase, wsPort)
    time.sleep(5)
    return wsAttn_modify



def channel_occ_simulate(index_of_occ = np.ones(64),wsAttn_for_flat = np.zeros(4000)):
    index_of_occ[0:3] = 0
    index_of_occ[53:] = 0
    wsFreq = np.arange(192.076, 196.076, 0.001)  # 1530-1561 nm
    wsAttn = np.ones(wsFreq.shape)*60
    wsPhase = np.zeros(wsFreq.shape)
    wsPort = np.ones(wsFreq.shape)
    
    # 1-3 信道 54-64信道不可用
    for index_channel in range(3,53):
        if index_of_occ[index_channel] !=0 :
            channel_indx = index_channel+1
            wsFreq = np.arange(192.076, 196.076, 0.001)
            # channel_indx = 53
            center_fre = 196.0875 - (channel_indx-1) * 0.075
            wsFreq_channel_occ = np.arange(center_fre-0.075/2, center_fre+0.075/2, 0.001)
            wsFreq_channel_sig_occ = np.arange(center_fre-0.06391/2, center_fre+0.06391/2, 0.001)
            wsFreq_channel_sig_indx =(np.floor((wsFreq_channel_sig_occ-192.076)/0.001)).astype(int)
            wsAttn[wsFreq_channel_sig_indx] = wsAttn_for_flat[wsFreq_channel_sig_indx]
    ws_r = uploadProfile(ip_waveshaper, wsFreq, wsAttn, wsPhase, wsPort)
    return wsAttn