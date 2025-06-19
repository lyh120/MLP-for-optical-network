import numpy as np
import random
import torch
import scipy.stats
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os
import torch.nn as nn


class LipschitzNN(nn.Module):
    def __init__(self, L_lambda, L_const, input_dim=12, output_dim=5):
        super().__init__()
        
        # self.feature_extractor = nn.Sequential(
        #     nn.Linear(input_dim, 64),
        #     nn.BatchNorm1d(64),
        #     nn.GELU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(64, 32),
        #     nn.BatchNorm1d(32),
        #     nn.GELU(),
        #     nn.Dropout(0.2)
        # )
        # self.predictor = nn.Sequential(
        #     nn.Linear(32, 24),
        #     nn.GELU(),
        #     nn.Linear(24, output_dim)
        # )
        self.Net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, output_dim)
        )

        self.lip_lambda = L_lambda
        self.scaler_X = None
        self.scaler_y = None
        self.Lip_const = torch.FloatTensor(L_const)
    # def forward(self, x):
    #     features = self.feature_extractor(x)
    #     return self.predictor(features)
    def forward(self, x):
        return self.Net(x)

    def lip_loss(self, x, y_pred):
            # 获取标准差
        sigma_x = torch.FloatTensor(self.scaler_X.scale_)  # [12]
        sigma_y = torch.FloatTensor(self.scaler_y.scale_)  # [5]

        # 调整上界矩阵
        Lip_scaled = self.Lip_const * (sigma_x.reshape(1, -1) / sigma_y.reshape(-1, 1))
        batch_size = x.size(0)
        # 计算雅可比矩阵 [batch_size, 5, 12]
        jacobian = torch.zeros(batch_size, y_pred.size(1), x.size(1)).to(x.device)
        for i in range(y_pred.size(1)):  # 对每个输出维度
            gradients = torch.autograd.grad(
                outputs=y_pred[:, i].sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True
            )[0]  # [batch_size, 12]
            jacobian[:, i, :] = gradients
        
        # 取绝对值
        jacobian_abs = torch.abs(jacobian)  # [batch_size, 5, 12]
        # 将Lip_scaled扩展到batch维度
        Lip_expanded = Lip_scaled.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, 5, 12]
        # 计算超过上界的部分
        excess = (jacobian_abs - Lip_expanded).clamp(min=0)
        # 计算损失
        loss = torch.mean(excess**2)
        return loss

# def resolution_shape(max_gradient, settings, qs):
#     mg = np.array(max_gradient)
    
#     def constraint(x_tar):
#         for set, q in zip(settings, qs):
#             delta = abs(np.array(set) - x_tar)
#             degrade = np.dot(mg, delta)
#             q_pred = q - degrade
#             # qmin = min(q)
#             if(min(q_pred) > 10): ## safe
#                 return 1
#             else:                   ## Safe
#                 continue
#         return 0
#     return constraint



class ConstraintManager:
    """约束函数管理封装类"""
    def __init__(self, cons_func):
        self.cons_func = cons_func
        self.cached_cons = lru_cache(maxsize=10000)(self._cached_wrapper)
        
    def _cached_wrapper(self, x_tuple):
        """元组转numpy数组的缓存包装"""
        return self.cons_func(np.array(x_tuple))
    
    def __call__(self, x):
        """函数式接口"""
        return self.cached_cons(tuple(x))

def generate_best_solution(settings, qs, cons, bounds, model=None):
    """
    贝叶斯优化算法
    主要逻辑：
    1. 增加ConstraintManager解决约束函数作用域问题
    2. 补充scipy.stats.norm引用
    3. 优化梯度计算稳定性
    """
    # 超参数配置
    config = {
        'n_candidates': 50,
        'exploit_ratio': 0.3,
        'xi': 0.01,
        'max_retries': 1000,
        'gp_restarts': 5
    }
    
    # 约束函数封装
    constraint_mgr = ConstraintManager(cons)
    
    # 数据预处理
    X = np.array(settings) if len(settings) > 0 else np.empty((0, len(bounds)))
    y = np.mean(qs, axis=1) if len(qs) > 0 else np.array([])
    
    # 优化阶段路由
    if len(X) < 5:
        return _initial_phase(bounds, constraint_mgr, model, config)
    else:
        return _online_phase(X, y, bounds, constraint_mgr, model, config)

def _initial_phase(bounds, constraint_mgr, model, config):
    """初始阶段逻辑"""
    # 生成候选
    candidates = _latin_hypercube(bounds, config['n_candidates'])
    
    # 约束过滤
    valid_mask = _parallel_constraint_check(candidates, constraint_mgr)
    valid_candidates = candidates[valid_mask]
    
    # 回退机制
    if len(valid_candidates) == 0:
        return _fallback_solution(bounds, constraint_mgr, config['max_retries'])
    
    # 模型预测选择
    if model is not None:
        scaled = _scale_params(valid_candidates, bounds)
        scores = _model_predict(model, scaled)
        return valid_candidates[np.argmin(scores)]
    
    # 启发式选择
    return _heuristic_select(valid_candidates)

def _online_phase(X, y, bounds, constraint_mgr, model, config):
    """在线优化阶段"""
    # 初始化高斯过程
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5) + WhiteKernel(noise_level=0.1),
        n_restarts_optimizer=config['gp_restarts']
    ).fit(X, y)
    
    # 生成混合候选
    candidates = _hybrid_candidate_generation(bounds, constraint_mgr, model, gp, config)
    
    # 约束过滤
    valid_mask = _parallel_constraint_check(candidates, constraint_mgr)
    valid_candidates = candidates[valid_mask]
    
    if len(valid_candidates) == 0:
        return _fallback_solution(bounds, constraint_mgr, config['max_retries'])
    
    # 贝叶斯优化核心
    return _bayesian_optimize(gp, valid_candidates, y, bounds, constraint_mgr, config)

# 核心工具函数 ----------------------------------------------------

def _hybrid_candidate_generation(bounds, constraint_mgr, model, gp, config):
    """混合候选生成"""
    candidates = _latin_hypercube(bounds, config['n_candidates'])
    
    # 模型引导生成
    if model is not None:
        n_samples = config['n_candidates'] // 2
        guided = _model_guided_samples(model, bounds, constraint_mgr,n_samples)  # 传递 n_samples 和 constraint_mgr
        # 确保guided不是空数组
        if guided is not None and guided.size > 0:
            if guided.shape[1] == candidates.shape[1]:  # 确保维度匹配
                candidates = np.vstack([candidates, guided])
            else:
                print("错误：_model_guided_samples返回的数组维度与candidates不匹配。")
        else:
            print("警告：_model_guided_samples返回了空数组。")
    
    # GP引导生成
    gp_guided = []
    for _ in range(10):  # 增加迭代次数
        x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        res = minimize(
            lambda x: -_expected_improvement(x, gp, np.min(gp.y_train_), config['xi']),
            x0,
            bounds=bounds,
            method='L-BFGS-B'
        )
        if res.success and constraint_mgr(res.x):
            gp_guided.append(res.x)
        else:
            print(f"GP引导生成的候选样本 {res.x} 不满足约束条件或优化失败")

    
    # 确保gp_guided不是空列表并且与candidates的维度匹配
    if gp_guided:
        gp_guided = np.array(gp_guided)
        if gp_guided.shape[1] == candidates.shape[1]:
            candidates = np.vstack([candidates, gp_guided])
        else:
            print("错误：gp_guided的维度与candidates不匹配。")
    else:
        print("警告：gp_guided是空列表。")
    
    return candidates


def _expected_improvement(x, gp, best, xi):
    """数值稳定的EI计算"""
    x = x.reshape(1, -1)
    mu, sigma = gp.predict(x, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    
    improvement = best - mu - xi
    z = improvement / sigma
    return improvement * scipy.stats.norm.cdf(z) + sigma * scipy.stats.norm.pdf(z)

def _bayesian_optimize(gp, candidates, y_history, bounds, constraint_mgr, config):
    """贝叶斯优化流程"""
    # 计算EI
    ei_values = np.array([_expected_improvement(x, gp, np.min(y_history), config['xi']) 
                         for x in candidates])
    
    # 选择候选
    ranked_indices = np.argsort(ei_values)[::-1][:int(config['n_candidates']*config['exploit_ratio'])]
    
    # 并行局部优化
    with ThreadPoolExecutor() as executor:
        optimized = list(executor.map(
            lambda x: _local_optimize(x, gp, bounds, constraint_mgr, config['xi']),
            candidates[ranked_indices]
        ))
    
    # 最终选择
    return min(optimized, key=lambda x: gp.predict([x])[0])

def _local_optimize(x0, gp, bounds, constraint_mgr, xi):
    """带约束的局部优化"""
    res = minimize(
        lambda x: -_expected_improvement(x, gp, np.min(gp.y_train_), xi),
        x0=x0,
        bounds=bounds,
        method='SLSQP',
        constraints={'type': 'ineq', 'fun': lambda x: float(constraint_mgr(x))},
        options={'maxiter': 50}
    )
    return res.x if res.success else x0

# 辅助工具函数 ----------------------------------------------------

def _latin_hypercube(bounds, n_samples):
    """改进的拉丁超立方采样"""
    dim = len(bounds)
    samples = np.zeros((n_samples, dim))
    for i in range(dim):
        lower, upper = bounds[i]
        samples[:, i] = np.random.uniform(lower, upper, n_samples)
    return samples

def _parallel_constraint_check(candidates, constraint_mgr):
    """并行约束检查"""
    with ThreadPoolExecutor() as executor:
        return np.array(list(executor.map(constraint_mgr, candidates)), dtype=bool)

def _fallback_solution(bounds, constraint_mgr, max_retries):
    """安全回退解"""
    for _ in range(max_retries):
        candidate = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        if constraint_mgr(candidate):
            return candidate
    return np.array([b[0] for b in bounds])

def _model_predict(model, inputs):
    """批量模型预测"""
    with torch.no_grad():
        return model(torch.FloatTensor(inputs)).numpy().flatten()

def _scale_params(x, bounds):
    """参数归一化"""
    return np.array([(x[:,i] - b[0])/(b[1]-b[0]) for i, b in enumerate(bounds)]).T

def _heuristic_select(candidates):
    """稳健启发式选择"""
    center = np.median(candidates, axis=0)
    dists = np.linalg.norm(candidates - center, axis=1)
    return candidates[np.argmin(dists)]

def _model_guided_samples(model, bounds, constraint_mgr, n_samples):
    candidates = []
    for _ in range(n_samples):
        x = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        scaled = _scale_params(x.reshape(1, -1), bounds)[0]
        x_tensor = torch.FloatTensor(scaled).requires_grad_(True)
        
        # 梯度计算
        pred = model(x_tensor)
        pred_sum = pred.sum()
        pred_sum.backward()
        grad = x_tensor.grad.numpy()
        
        # 梯度方向更新
        step = 0.1 * (np.array([b[1] - b[0] for b in bounds]) * np.sign(grad))
        new_x = np.clip(x + step, [b[0] for b in bounds], [b[1] for b in bounds])
        
        if constraint_mgr(new_x):
            candidates.append(new_x)
        else:
            print(f"候选样本 {new_x} 不满足约束条件")
    
    if not candidates:
        print("没有找到满足约束条件的候选样本")
    return np.array(candidates)


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

if __name__=='__main__':
    #模拟参数定义
    x0 = [16,25,18,10,25,18,-1,-1,-1,-1,-1,-1]
    #x0= [ 18.82236206,28.99922894 ,18.69651744, 14.56841873 ,27.7090383,  21.67708612 ,-0.13094517 ,-0.69131987 ,-0.38605406, -0.74020146 ,-1.20420734, -1.66281231]
    patience=5
    Lip_data = np.load('./Lip_GN_5wave_0319.npz')
    Lip = Lip_data['mgn_list'][-1]
    L_const = Lip
    seed=42
    model_path = 'enhanced_model.pth'
    Lip_lambda=0.2

    #  模型加载与初始化
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到预训练模型 {model_path}")

    # 初始化当前模型
    model =  model = LipschitzNN(L_lambda=Lip_lambda, L_const=Lip)
    
    try:
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 加载模型参数（忽略scaler相关键值）
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # 加载标准化器
        model.scaler_X = checkpoint['scaler_X']
        model.scaler_y = checkpoint['scaler_y']
        
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}") from e

    print("\n=== 成功加载增强模型 ===")

    
    
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
    loaded_data = np.load("./Lip_GN_5wave_0319.npz")
    setting_list = loaded_data['settings'].reshape(-1, 12)[5:10,:]
    q_list = loaded_data['qs'].reshape(-1, 5)[5:10,:]
    # 初始化
    settings = []
    qs = []
    settings = setting_list
    qs = q_list

    cons = resolution_shape(L_const, settings, qs) # 安全空间限制
    print(cons)
        ##### 在安全空间内，利用模型找到最优配置
    new_solution = generate_best_solution(settings, qs, cons, bounds, model) # TODO 
    print(new_solution)