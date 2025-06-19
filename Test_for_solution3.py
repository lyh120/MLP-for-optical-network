
import numpy as np
import torch
import torch.nn as nn
import os
import random
from scipy.stats import qmc  # 新增LHS采样库

class LipschitzNN(nn.Module):
    # 保持原有模型结构不变
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
    
    # 保持原有lip_loss实现不变
    def lip_loss(self, x, y_pred):
        sigma_x = torch.FloatTensor(self.scaler_X.scale_)
        sigma_y = torch.FloatTensor(self.scaler_y.scale_)
        Lip_scaled = self.Lip_const * (sigma_x.reshape(1, -1) / sigma_y.reshape(-1, 1))
        batch_size = x.size(0)
        
        jacobian = torch.zeros(batch_size, y_pred.size(1), x.size(1)).to(x.device)
        for i in range(y_pred.size(1)):
            gradients = torch.autograd.grad(
                outputs=y_pred[:, i].sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True
            )[0]
            jacobian[:, i, :] = gradients
        
        jacobian_abs = torch.abs(jacobian)
        Lip_expanded = Lip_scaled.unsqueeze(0).expand(batch_size, -1, -1)
        excess = (jacobian_abs - Lip_expanded).clamp(min=0)
        return torch.mean(excess**2)

def resolution_shape(max_gradient, settings, qs):
    # 保持原有约束函数不变
    mg = np.array(max_gradient)
    
    def constraint(x_tar):
        for set, q in zip(settings, qs):
            delta = abs(np.array(set) - x_tar)
            degrade = np.dot(mg, delta)
            q_pred = q - degrade
            if min(q_pred) > 10:
                return 1
        return 0
    return constraint

def generate_best_solution(settings, qs, cons, bounds, model, 
                          batch_size=100, noise_scale=0.05, noise_increment=0.1,
                          target_points=500, max_iters=100):
    """
    改进点：
    1. 使用混合采样策略(LHS + 高斯噪声)
    2. 动态扩展解空间
    3. 强制包含当前最优解
    """
    def _scale_params(x, bounds):
        x = np.array(x)
        x_norm = np.zeros_like(x, dtype=float)
        for i, (lb, ub) in enumerate(bounds):
            x_norm[..., i] = (x[..., i] - lb) / (ub - lb)
        return x_norm

    def _model_predict(model, inputs):
        with torch.no_grad():
            inputs_tensor = torch.FloatTensor(inputs)
            preds = model(inputs_tensor).sum(dim=1)
            return preds.numpy()

    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    dim = len(bounds)
    
    # 混合采样参数
    lhs_ratio = 0.7  # LHS采样比例
    current_point = settings[-1]  # 当前最优解

    satisfied_points = []
    for _ in range(max_iters):
        all_points = []
        
        # 围绕每个历史点生成候选
        for center in settings:
            # 动态计算局部范围
            local_lb = np.clip(center - noise_scale*(upper_bounds - lower_bounds), lower_bounds, upper_bounds)
            local_ub = np.clip(center + noise_scale*(upper_bounds - lower_bounds), lower_bounds, upper_bounds)
            
            # LHS采样
            lhs_samples = qmc.LatinHypercube(d=dim).random(n=int(batch_size*lhs_ratio))
            lhs_samples = qmc.scale(lhs_samples, local_lb, local_ub)
            
            # 高斯采样
            gauss_samples = center + np.random.normal(
                scale=noise_scale, 
                size=(int(batch_size*(1-lhs_ratio)), dim)
            ) * (upper_bounds - lower_bounds)
            
            # 合并并处理
            candidates = np.vstack([lhs_samples, gauss_samples])
            candidates = np.clip(candidates, lower_bounds, upper_bounds)
            candidates = np.round(candidates, 1)  # 保持一位小数
            all_points.extend(candidates)

        # 强制包含当前解
        all_points.append(current_point)
        
        # 过滤满足约束的点
        new_valid = [pt for pt in all_points if cons(pt) == 1]
        satisfied_points.extend(new_valid)
        
        # 提前终止
        if len(satisfied_points) >= target_points:
            break

    # 候选点不足时的处理
    if not satisfied_points:
        print("无有效候选，返回当前解")
        return current_point
    
    # 评估候选点（包含当前解）
    candidates_scaled = _scale_params(satisfied_points, bounds)
    scores = _model_predict(model, candidates_scaled)
    best_idx = np.argmax(scores)
    best_solution = satisfied_points[best_idx]
    
    # 比较当前解与新解
    current_score = _model_predict(model, _scale_params([current_point], bounds))[0]
    if scores[best_idx] > current_score:
        print(f"找到更优解：{scores[best_idx]:.2f} > {current_score:.2f}")
        return best_solution
    else:
        # 动态扩展搜索范围
        new_noise = noise_scale * (1 + noise_increment)
        print(f"未找到改进，扩大噪声至：{new_noise:.3f}")
        return generate_best_solution(
            settings=np.vstack([settings, best_solution]),  # 扩展解空间
            qs=np.vstack([qs, qs[-1]]),  # 假设新解的q值与当前相同
            cons=cons,
            bounds=bounds,
            model=model,
            noise_scale=new_noise,
            noise_increment=noise_increment,
            target_points=target_points,
            max_iters=max_iters
        )

# 模拟参数和初始化（保持原有部分不变）
if __name__ == "__main__":
    # 参数初始化
    x0 = [18.8, 29.0, 18.7, 14.6, 27.7, 21.7, -0.1, -0.7, -0.4, -0.7, -1.2, -1.7]
    bounds = [(14,23), (22,30), (16,22), (8,15), (22,30), (16,22)] + [(-2,0)]*6
    
    # 加载模型和数据
    model = LipschitzNN(L_lambda=0.2, L_const=np.load('./Lip_GN_5wave_0319.npz')['mgn_list'][-1])
    checkpoint = torch.load('enhanced_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.scaler_X = checkpoint['scaler_X']
    model.scaler_y = checkpoint['scaler_y']
    
    # 初始化设置
    settings = np.array([x0])
    qs = np.array([[85, 90, 88, 92, 95]])  # 示例质量值
    
    # 生成约束函数
    cons = resolution_shape(
        max_gradient=model.Lip_const.numpy(),
        settings=settings,
        qs=qs
    )
    
    # 运行优化
    best = generate_best_solution(
        settings=settings,
        qs=qs,
        cons=cons,
        bounds=bounds,
        model=model,
        noise_scale=0.1,
        noise_increment=0.2,
        target_points=500
    )
    print("最终最优解:", best)
