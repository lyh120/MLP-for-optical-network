"""
Yokogawa AQ637X OSA sample program (SOCKET)
"""
import socket
import numpy as np
import scipy
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy import integrate
import os
import time
from GlobalControl import GlobalControl
try:
    logger = GlobalControl.logger
except:
    logger = GlobalControl.init_logger()
    logger.warning('GlobalControl中logger未定义！')

logger.info('osa.py imported.')


def dbm2w(dbm):
    return 10.0 ** (dbm / 10.0) *1e-3


def w2dbm(w):
    return 10 * np.log10(w / 1e-3)


def Read_OSA(s, BUFFER_SIZE):
    # data = s.recv(BUFFER_SIZE)
    data = b''
    while True:
        try:
            recv_data = s.recv(BUFFER_SIZE)
            if len(recv_data) > 0:
                data += recv_data
            else:
                break
        except Exception as e:
            # print('socket receiving data error! |'+str(e))
            # print('read: ', len(data), ', data: ', data)
            return data.decode('ascii')


def Write_OSA(s, MESSAGE, BUFFER_SIZE=None):
    N = s.send(MESSAGE.encode())
    # print('send:', N, ', data: ', MESSAGE.encode())
    return N

def str2num(string):
    num = []
    string = string.split(',')
    for i in string:
        num.append(float(i))
    return np.array(num)

class OSA(object):
    def __init__(self, start_end, file_name, file_path) -> None:
        self.TCP_IP_yokogawa = '192.168.1.11'
        self.TCP_PORT = 10001
        self.BUFFER_SIZE = 10000        #这个可以设大点，反正没啥用
        self.Name = 'anonymous'
        self.passwd = ''
        self.start_freq = start_end[0]
        self.end_freq = start_end[1]
        self.Filename = file_name
        self.SaveDir = file_path

    def OpenTCPsocket(self):
        # print(TCP_IP, TCP_PORT, BUFFER_SIZE)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        try:
            s.connect((self.TCP_IP_yokogawa, self.TCP_PORT))
        except Exception:
            raise Exception('socket open error.')

        logger.info('OSA open: complete')
        return s

    def CloseTCPSocket(self, s):
        s.close()

    def Open_OSA(self, OSA):
        OpeningYokogawa_str = "OPEN \"anonymous\"\r\n"
        Write_OSA(OSA, OpeningYokogawa_str)
        Read_OSA(OSA, self.BUFFER_SIZE)
        Write_OSA(OSA, self.passwd+"\r\n")
        Read_OSA(OSA, self.BUFFER_SIZE)

        Write_OSA(OSA, "*IDN?\r\n")
        test_str = Read_OSA(OSA, self.BUFFER_SIZE)

        if test_str == '' or test_str == None:
            raise Exception("OSA Not connected!")

        logger.info('OSA connected')
    
    def Set_OSA(self, OSA):
        # 
        Write_OSA(OSA, ":SENSE:WAVELENGTH:START " +  str(self.start_freq) + "THZ\r\n")
        Write_OSA(OSA, ":SENSE:WAVELENGTH:STOP " +  str(self.end_freq) + "THZ\r\n")
        # 窗宽度
        Write_OSA(OSA, ":SENSE:BANDWIDTH:RESOLUTION 5GHZ" + "\r\n")
        # 采样点个数 间隔=span/数量 与宽度无关
        # Write_OSA(OSA, ":SENSE:SWEEP:POINTS?" + "\r\n")  # Query
        sampling_number =int(np.ceil( (self.end_freq-self.start_freq)/0.0025))
        Write_OSA(OSA, ":SENSE:SWEEP:POINTS "+ str(sampling_number) + "\r\n")  # Set
        
        Write_OSA(OSA, ":SENSE:SENSE NORMAL\r\n")
        Write_OSA(OSA, ":INITIATE:SMODE REPEAT\r\n")
    
    def collect_data(self, OSA, trace_str = "TRA"):
        Write_OSA(OSA, ":TRACE:ACTive " + trace_str + "\r\n")

        Write_OSA(OSA, ":TRACE:X? " + trace_str + "\r\n")
        f = Read_OSA(OSA, self.BUFFER_SIZE)

        # Write_OSA(OSA, ":TRACE:Y? " + trace_str + "\r\n")
        # y = Read_OSA(OSA, self.BUFFER_SIZE)
        # 是换算到0.1nm内功率值 即 x/resolution*12.5G
        Write_OSA(OSA, ":TRACE:Y:PDENsity? " + trace_str + ", 0.1NM\r\n")
        psd = Read_OSA(OSA, self.BUFFER_SIZE)
        # data process
        f = str2num(f[:-2])
        psd = str2num(psd[:-2])
        # nm -> THz
        c = 299792458
        f = c/f/10**12    

        return psd, f

    def OSA_data_save(self, plot=0, name=None, trace_str="TRA"):
        OSA = self.OpenTCPsocket()
        # OPEN
        self.Open_OSA(OSA)
        self.Set_OSA(OSA)
        time.sleep(5)
        psd, f = self.collect_data(OSA, trace_str=trace_str)
        # data save
        if name is None:
            name = self.Filename 
        FileName = self.SaveDir + name + '.npz'
        np.savez_compressed(FileName, psd=psd, f=f)
        # close OSA
        self.CloseTCPSocket(OSA)
        if plot:
            # sns.set_theme(style="darkgrid")
            fig, axs = plt.subplots(1, 1, figsize=(8, 6), edgecolor = '#282C34')
            fig.set_facecolor('#282C34')
            fig.set_edgecolor('#282C34')
            fig.suptitle('OSA PSD', color = '#B4B6BD', fontsize = 'xx-large')
            axs.set_facecolor('k')
            axs.plot(f, psd, '-o', c='gold')
            label_c = '#B4B6BD'
            axs.set_xlabel('Frequency (GHz)', color = label_c, fontsize = 'large', fontstyle = 'italic')
            axs.set_ylabel('PSD (dB)', color = label_c, fontsize = 'large', fontstyle = 'italic')
            axs.spines['left'].set_color(label_c)
            axs.spines['right'].set_color(label_c)
            axs.spines['top'].set_color(label_c)
            axs.spines['bottom'].set_color(label_c)
            axs.tick_params(labelcolor= label_c, labelsize = 'large')
            fig.savefig(self.SaveDir+'OSA_PSD_'+name+'.png')
            plt.show()
        logger.info('OSA save Done!')
        return psd,f


def psd_process(f_range, psd=None, freq=None, psd_filename=None):
    if psd_filename is not None:
        logger.debug('Load psd from file.')
        psdfile =  np.load(psd_filename)
        psd = psdfile.f.psd
        freq = psdfile.f.f

    channel_num = int((f_range[1]-f_range[0])/0.02)
    psd_out = np.zeros(channel_num)
    freq_out = np.arange(f_range[0], f_range[1]-1e-3, 0.02) + 0.01
    for i in range(channel_num):
        start_freq_temp = f_range[0] + i*0.02 - 1e-3
        end_freq_temp = f_range[0] + (i+1)*0.02 - 1e-3
        psd_temp = psd[(freq>start_freq_temp) * (freq<end_freq_temp)]
        psd_out[i] = w2dbm(np.mean(dbm2w(psd_temp)))
    
    return psd_out, freq_out


def measure_fiberout_psd_osnr(ind=0,f_range = [187, 196.5]):
    CURRENT_PATH = os.getcwd()
    save_path = CURRENT_PATH 
    # osa_obj_name = 'fiberout_psd_woRA'
    osa_obj_name = f'fiberout_psd_wRA_{ind}'
    #f_range = [186, 196.5]
    osa_obj = OSA(f_range, osa_obj_name, save_path)
    psd, f= osa_obj.OSA_data_save(plot=0, trace_str="TRA")

    # print(f'before psd process: shape(psd)={np.shape(psd)}')
    # plt.plot(f, psd)
    
    # psd, f = psd_process_200G(f_range, psd, f)
    # f = np.round(f, 3)
    
    plt.plot(f, psd)
    plt.show(block=False)
    # print(f'after psd process: shape(psd)={np.shape(psd)}')
    scipy.io.savemat(save_path + osa_obj_name + '.mat',
                    mdict={'psd': psd, 'f': f})
    return psd, f



def measure_fiberout_psd(ind=0,f_range = [187, 196.5]):
    CURRENT_PATH = os.getcwd()
    save_path = CURRENT_PATH 
    # osa_obj_name = 'fiberout_psd_woRA'
    osa_obj_name = f'fiberout_psd_wRA_{ind}'
    #f_range = [186, 196.5]
    osa_obj = OSA(f_range, osa_obj_name, save_path)
    psd, f= osa_obj.OSA_data_save(plot=0, trace_str="TRA")

    # print(f'before psd process: shape(psd)={np.shape(psd)}')
    # plt.plot(f, psd)
    
    # psd, f = psd_process_200G(f_range, psd, f)
    # f = np.round(f, 3)
    
    plt.plot(f, psd)
    plt.show(block=False)
    # print(f'after psd process: shape(psd)={np.shape(psd)}')
    scipy.io.savemat(save_path + osa_obj_name + '.mat',
                    mdict={'psd': psd, 'f': f})
    return psd, f

if __name__ == '__main__':
    # psd, f = measure_ase_shape()
    psd, f = measure_fiberout_psd(0,[186, 196.5])
    print(np.shape(f))
    plt.plot(f, psd)
    
    plt.show()
