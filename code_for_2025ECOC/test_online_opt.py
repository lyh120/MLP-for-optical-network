import numpy as np
import time, random
from utilities.utils import apply_parallel, query_parallel, query_setting #初始化会消耗较多时间(10s左右)

def resolution_shape(max_gradient, settings, qs):
    mg = np.array(max_gradient)
    
    def constraint(x_tar):
        for set, q in zip(settings, qs):
            delta = abs(np.array(set) - x_tar)
            degrade = np.dot(mg, delta)
            q_pred = q - degrade
            # qmin = min(q)
            if(min(q_pred) > 10): ## safe
                return 1
            else:                   ## Safe
                continue
        return 0
    return constraint

def online_opt(x0, iter_num=20, seed=42, early_stop=True, patience=5, max_gradient=np.ones([5,12]), save_name = 'test'):
    ## Hill Climbing
    # 设置优化参数
    bounds = [(14, 23), (22, 30), (16, 22), (8, 15), (22,30), (16, 22),(-2, 0), (-2, 0), (-2, 0), (-2, 0), (-2, 0), (-2, 0)]
    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]

    random.seed(seed)
    np.random.seed(seed)
    
    # Early stop 相关变量
    best_solution = x0
    no_improvement_count = 0
    
    # 初始化
    settings = []
    qs = []
    apply_parallel(x0)
    settings.append(np.array(x0))
    q0 = query_parallel(avg_times=3)
    qs.append(q0)
    # best_fitness = (np.mean(q0) + np.min(q0))
    best_fitness = np.mean(q0)
    best_Q = np.array(q0)
    L_const = max_gradient 
    start = time.time()
    
    for index in range(iter_num):
        cons = resolution_shape(L_const, settings, qs) # 安全空间限制
        
        ##### 在安全空间内，利用模型找到最优配置
        new_solution = generate_best_solution(settings, qs, cons, bounds, model) # TODO 
        
        apply_parallel(new_solution) # 部署配置
        current_Q = query_parallel() # 查询配置
        settings.extend(new_solution)
        qs.extend(current_Q)
        
        #### Online training for the model
        # TODO
        
        ##### Show process
        current_solution = new_solution
        current_fitness_value = np.mean(current_Q)
        print(f"Iteration {index}: Current Solution = {current_solution}")
        print(f"Current Fitness = {current_fitness_value}, Current Q={current_Q}")
        print()
        
        ## Best x identification
        if current_fitness_value < best_fitness:
            best_solution = current_solution
            best_fitness = current_fitness_value
            best_Q = current_Q
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        ## Save
        np.savez(save_name+'.npz', qs=np.array(qs), settings=np.array(settings), best_setting = np.array(best_solution), 
            best_q = np.squeeze(best_Q))
        print(f"No improvement count = {no_improvement_count}")
        print(f"Best Solution = {best_solution}, Best Fitness = {best_fitness}, Best Q={best_Q}")
        # Early stopping
        if early_stop and (no_improvement_count >= patience):
            print(f"Early stopping at Iteration {index} due to no improvement in {patience} generations.")
            break
        
    end = time.time()
    print(f"Optimal Solution = {best_solution}, Optimal Fitness = {best_fitness}, Best Q={best_Q}")
    print()
    print("Time:", (end - start))
    
if __name__=='__main__':
    Lip_data = np.load('./Lip_GN_5wave_0319.npz')
    Lip = Lip_data['mgn_list'][-1]
    x = [16,25,18,10,25,18,-1,-1,-1,-1,-1,-1]
    online_opt(x0=x, iter_num=40, seed=42, early_stop=True, patience=5, max_gradient= Lip , save_name='test')