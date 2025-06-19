import numpy as np
import torch
import torch.nn as nn
import os
import random
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
class LipschitzNN(nn.Module):
    def __init__(self, L_lambda, L_const, input_dim=12, output_dim=5):
        super().__init__()
        
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
def generate_best_solution(settings, qs, cons, bounds, model, 
                           batch_size=100, noise_scale=0.05, 
                           target_points=500, max_iters=100):
    """
    在已有的 settings 周围不断随机采样点，过滤掉不满足 cons 的点；
    当满足条件的点数量达到 target_points 或者迭代次数超过 max_iters 时停止。
    最后在所有满足条件的点中，用给定的 model 评估并选出预测值最小的那个点作为 “最优解”。
    随着在线数据的增加，通过将新数据添加到settings和qs列表中来更新解空间。
    新的候选解将在更大的空间内生成，从而实现了解空间的扩大。

    参数:
    -------
    settings: array-like, 形如 (N, D)，已有的 N 组历史参数, 每个参数 D 维
    
    qs: array-like, 与 settings 同长度

    cons: callable，约束函数, 输入一个参数向量 x, 如果满足则返回 1, 否则返回 0

    bounds: list of tuple，每个维度的取值范围, [(lb1, ub1), (lb2, ub2), ...]

    model: torch.nn.Module，用于对候选解进行预测的模型, 假设输出越大越好

    batch_size: int，每个 setting 周围一次性采样的点数

    noise_scale: float，采样噪声的缩放系数, 数值越大, 搜索范围越“大”

    target_points: int，期望至少找到多少个满足 cons 的候选点

    max_iters: int，最大迭代轮数, 如果在此轮数内都找不到足够多的候选点，则停止

    返回值:
    -------
    best_solution: np.ndarray，在所有满足约束的点中, 根据模型预测值最大的解
    """
    def _scale_params(x, bounds):
        """将 x 按照 bounds 归一化到 [0, 1]"""
        x = np.array(x)
        x_norm = np.zeros_like(x, dtype=float)
        for i, (lb, ub) in enumerate(bounds):
            x_norm[..., i] = (x[..., i] - lb) / (ub - lb)
        return x_norm

    def _model_predict(model, inputs):
        """将 inputs (N, D) 批量送入模型, 获取预测值 (N,)"""
        with torch.no_grad():
            inputs_tensor = torch.FloatTensor(inputs)
            preds = model(inputs_tensor)  # 模型输出形状为 (N, 5)
            # 取每个样本五个输出的总和作为预测值
            preds_sum = preds.sum(dim=1) 
            return preds_sum.numpy()

    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    satisfied_points = []
    total_generated_points = 0
    dim = len(bounds)

    for num in range(max_iters):
        all_points = []
        for setting in settings:
            noise = np.random.normal(0, noise_scale, (batch_size, dim))
            candidates = setting + noise * (upper_bounds - lower_bounds)
            candidates = np.clip(candidates, lower_bounds, upper_bounds)
            # 四舍五入到一位小数
            candidates = np.round(candidates, decimals=1)
            all_points.extend(candidates)

        all_points = np.array(all_points)
        total_generated_points += len(all_points)
        new_satisfied_points = [pt for pt in all_points if cons(pt) == 1]
        satisfied_points.extend(new_satisfied_points)

        print(f"iter = {num}: 当前满足条件的点数: {len(satisfied_points)}, "
              f"总生成点数: {total_generated_points}")

        if len(satisfied_points) >= target_points:
            break

    if len(satisfied_points) == 0:
        print("没有找到任何满足约束的候选点，返回下界作为 fallback")
        return np.array(lower_bounds)

    satisfied_points = np.array(satisfied_points)
    satisfied_points_scaled = _scale_params(satisfied_points, bounds)
    predictions = _model_predict(model, satisfied_points_scaled)
    best_idx = np.argmax(predictions)
    best_solution = satisfied_points[best_idx]
    return best_solution


#模拟参数定义
#x0 = [16,25,18,10,25,18,-1,-1,-1,-1,-1,-1]
x0= [ 18.82236206,28.99922894 ,18.69651744, 14.56841873 ,27.7090383,  21.67708612 ,-0.13094517 ,-0.69131987 ,-0.38605406, -0.74020146 ,-1.20420734, -1.66281231]
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

########
dataset_settings = np.round(loaded_data['settings'],1)
setting_list = dataset_settings.reshape(-1, 12)[2,:]# 加载数据,一个点
q_list = loaded_data['qs'].reshape(-1, 5)[2,:]
# 初始化
settings = []
qs = []
settings = setting_list.reshape(1,-1)
qs = q_list.reshape(1,-1)
print("初始设置:", settings)
print("初始质量:", qs)

cons = resolution_shape(L_const, settings, qs) # 安全空间限制

best_sol = generate_best_solution(settings, qs, cons, bounds, model,
                                  batch_size=100,
                                  noise_scale=0.05,
                                  target_points=500,
                                  max_iters=1000)
print("最终得到的最优解:", best_sol)
