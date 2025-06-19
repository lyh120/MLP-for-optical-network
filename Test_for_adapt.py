import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import os

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
loaded_data = np.load("./Lip_GN_5wave_0319.npz")
setting_list = loaded_data['settings'].reshape(-1, 12)[2:6,:]
q_list = loaded_data['qs'].reshape(-1, 5)[2:6,:]
#print(setting_list, q_list)
settings = setting_list
qs = q_list
# 自适应在线微调
def adaptive_online_finetuning(model=None, 
                              num_samples=5,   # 建议至少5个样本
                              epochs=50,       # 增加训练轮次
                              lr=1e-4,        # 调整初始学习率
                              Lip_lambda=0.2,# Lipschitz约束系数
                              L_const=np.ones([5,12]), # Lipschitz约束矩阵
                              setting_list = None,
                              q_list = None,
                              save_path = 'enhanced_model.pth'
                              ):
    setting_list = setting_list.reshape(-1, 12)
    q_list = q_list.reshape(-1, 5)
    #print(setting_list, q_list)
    # 数据增强函数
    def sliding_window_augmentation(data, window_size=3):
        """滑动窗口数据增强"""
        augmented = []
        for i in range(len(data)-window_size+1):
            window = data[i:i+window_size]
            # 添加均值扰动
            augmented.append(np.mean(window, axis=0))
            # 添加噪声扰动
            augmented.append(window[1] + np.random.normal(0, 0.01, window[1].shape))
        return np.array(augmented)
    
    # 获取原始数据并增强
    #start_idx = random.randint(1, 50)  # 起始样本索引
    raw_configs = setting_list #[start_idx:start_idx+num_samples*2]  # 原始数据
    raw_q = q_list #[start_idx:start_idx+num_samples*2]
    print(raw_configs, raw_q)
    aug_configs = sliding_window_augmentation(raw_configs)
    aug_q = sliding_window_augmentation(raw_q)


    # 合并数据集
    real_configs = np.vstack([raw_configs, aug_configs])
    real_q = np.vstack([raw_q, aug_q])

    # 标准化处理
    real_configs_scaled = model.scaler_X.transform(real_configs)
    real_q_scaled = model.scaler_y.transform(real_q)

    # 创建数据加载器
    dataset = TensorDataset(
        torch.FloatTensor(real_configs_scaled),
        torch.FloatTensor(real_q_scaled)
    )
    loader = DataLoader(dataset, 
                      batch_size=min(8, len(dataset)),  # 动态批次大小
                      shuffle=True,
                      pin_memory=True)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    
    # 带热重启的余弦退火调度
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,        # 每10个epoch重启
        T_mult=1,
        eta_min=1e-6
    )
    
    # 早停机制
    best_loss = float('inf')
    patience = 8
    patience_counter = 0
    # 微调模型
    print(f"\n=== 开始微调 (增强后样本数: {len(dataset)}) ===")
    losses = []
    grad_magnitudes = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_grad = 0.0
        
        for x, y in loader:
            x.requires_grad_(True)  # 启用梯度计算
            
            optimizer.zero_grad()
            
            # 前向传播
            pred = model(x)
            
            # 复合损失计算
            mse_loss = nn.MSELoss()(pred, y)
            lip_loss = model.lip_loss(x, pred)
            total_loss = mse_loss + model.lip_lambda * lip_loss
            
            # 反向传播
            total_loss.backward()
            
            # 梯度监控
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),  # 只监控模型参数
                max_norm=1.0,  #可降低阈值
                norm_type=2
            )
            
            # 记录梯度量级
            total_grad += grad_norm.item()
            
            optimizer.step()
            epoch_loss += total_loss.item()
        
        # 更新学习率
        scheduler.step()
        
        # 计算指标
        avg_loss = epoch_loss / len(loader)
        avg_grad = total_grad / len(loader)
        losses.append(avg_loss)
        grad_magnitudes.append(avg_grad)
        
        # 早停判断
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f"fine_tuned_best.pth")
        else:
            patience_counter += 1
        
        # 打印训练信息
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d}/{epochs} | "
              f"Loss: {avg_loss:.3e} | "
              f"Grad: {avg_grad:.2f} | "
              f"LR: {current_lr:.2e} | "
              f"Patience: {patience_counter}/{patience}")
        
        # 早停触发
        if patience_counter >= patience:
            print(f"早停触发，最佳loss: {best_loss:.4e}")
            break
     # 保存模型和标准化器的状态
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_X': model.scaler_X,
        'scaler_y': model.scaler_y
    }
    torch.save(checkpoint, f"enhanced_model.pth")    
    # 返回模型
    return model
model = adaptive_online_finetuning(model= model,
                              num_samples=3,   # 建议至少5个样本
                              epochs=50,       # 增加训练轮次
                              lr=1e-4,        # 调整初始学习率
                              Lip_lambda=0.2, # Lipschitz约束系数
                              L_const=L_const, # Lipschitz约束矩阵
                              setting_list = settings,
                              q_list = qs,
                              save_path = 'fine_model.pth'
                              )