import sys
sys.path.append('./utilities')
from osa import *
from loadfilter import *
import numpy
from scipy import interpolate
import os
from tencent import *
from BaianCEDFA import *
from BaianLEDFA import *

def ini_C_OA():
    BA, _,  iA2B, iB2A, _, PA=ini_TencentOA()
    slot2, slot3, slot6 = ini_BainanC_OA()
    return BA, iA2B, iB2A, slot2, slot3, slot6, PA
