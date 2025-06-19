import numpy as np
import pandas as pd
from scipy.stats import qmc
from utilities.utils import apply_parallel, query_parallel, query_all_input_power, query_all_output_power, query_all_ocm
import time

def function(x):
    if(x.ndim==1):
        for i in range(len(x)):
            x[i] = round(x[i],1)
        apply_parallel(x)
        q = query_parallel(avg_times=3)
        return np.array(q)

    elif x.ndim == 2:
        results = []
        Q = []
        for row in x:
            for i in range(len(row)):
                row[i] = round(row[i], 1)
            apply_parallel(row)
            q = query_parallel(avg_times=3)
            Q.append(q)
            print("Setting:", row)
            print("Q factor:", q)
        return np.array(Q)

if __name__=='__main__':
    
    # 定义边界
    bounds = [(14, 23), (22, 30), (16, 22), (8, 15), (22,30), (16, 22),(-2, 0), (-2, 0), (-2, 0), (-2, 0), (-2, 0), (-2, 0)]

    # 拉丁超立方采样
    sampler = qmc.LatinHypercube(d=len(bounds))
    sample = sampler.random(n=500)

    # 将采样点转换到指定范围内
    lower_bounds, upper_bounds = np.transpose(bounds)
    sample_scaled = qmc.scale(sample, lower_bounds, upper_bounds)
    sample_scaled = np.round(sample_scaled, 1)
    
    setting_list = []
    q_list = []
    Pin_list = []
    Pout_list = []
    psd_list = []
    f_list = []
    
    for index, point in enumerate(sample_scaled):
        print(f"Processing iteration: {index + 1}/{len(sample_scaled)}") 
        apply_parallel(point)
        # time.sleep()
        q = query_parallel(avg_times=3)
        Pin = query_all_input_power()
        Pout = query_all_output_power()
        psd, f = query_all_ocm()
        print(q)
        print(Pin)
        print(Pout)
        
        setting_list.append(point)
        q_list.append(q)
        Pin_list.append(Pin)
        Pout_list.append(Pout)
        psd_list.append(psd)
        f_list.append(f[0])
        
        np.savez('./ecoc_data/dataset_5wave_500sample0317.npz', setting_list=np.array(setting_list), q_list=np.array(q_list),
                Pin_list=np.array(Pin_list), Pout_list=np.array(Pout_list),  psd_list=np.array(psd_list), f_list = np.array(f_list))

    # # 初始化CSV文件
    # columns = [f'input_{i+1}' for i in range(len(bounds))] + [f'output_{i+1}' for i in range(2)]
    # df = pd.DataFrame(columns=columns)
    # df.to_csv('brute_search_2wave.csv', index=False)

    # # 逐步生成点并保存结果
    # for point in sample_scaled:
    #     result = function(point)
    #     data = np.hstack((point, result)).reshape(1, -1)
    #     df = pd.DataFrame(data, columns=columns)
    #     df.to_csv('brute_search_2wave.csv', mode='a', header=False, index=False)

    print("Sampling and saving completed.")