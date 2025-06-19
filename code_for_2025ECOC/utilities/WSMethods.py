import requests
import numpy as np
import json

def createWspString(wsFreq, wsAttn, wsPhase, wsPort):
    wsAttn[np.isnan(wsAttn)] = 60
    wsPhase[np.isnan(wsPhase)] = 0
    wsAttn[wsAttn>60] = 60
    wsAttn[wsAttn<=0] = 0
    wspString = ''
    for i in range(len(wsFreq)):
        wspString = '%s%.4f\t%.4f\t%.4f\t%d\n' % (wspString, wsFreq[i], wsAttn[i], wsPhase[i], wsPort[i])
    return wspString


def uploadProfile(ip, wsFreq, wsAttn, wsPhase, wsPort, timeout=10):
    data = {'type': 'wsp', 'wsp': createWspString(wsFreq, wsAttn, wsPhase, wsPort)}
    r = requests.post('http://'+ip+'/waveshaper/loadprofile', json.dumps(data), timeout=timeout)
    return r

