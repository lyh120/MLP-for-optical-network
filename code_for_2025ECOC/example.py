from utilities.utils import apply_parallel, query_parallel, query_setting  # import这些控制函数初始化会消耗较多时间(10s左右)
import numpy as np

if __name__=='__main__':
    ### 配置函数
    bounds = np.array([(14, 23), (22, 30), (16, 22), (8, 15), (22,30), (16, 22),(-2, 0), (-2, 0), (-2, 0), (-2, 0), (-2, 0), (-2, 0)])
    x0 = [17,24,19,11,23,17,-1.5,-1,-1,-1,-1,-1]
    x0 = np.clip(x0, bounds[:, 0], bounds[:, 1])
    x0 = np.round(x0, 1)
    apply_parallel(x0)  # 部署的配置有上下界，精度为1位小数。
    
    #### Q-factor查询函数
    qs = query_parallel()
    print(f'Q-factor: {qs}')
    
    #### 配置查询函数
    settings = query_setting()
    print(f'Settings: {settings}')