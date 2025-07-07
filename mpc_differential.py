import numpy as np
import cvxpy as cp
import math
from matplotlib import pyplot as plt

# ========== 参数调整 ==========
N = 8  # 减小预测步长
T = 0.2  # 增大控制周期
WHEELBASE = 0.5
MAX_SPEED = 1.0  # 降低最大速度
MAX_OMEGA = 1.0  # 降低最大角速度

# 权重矩阵（增加稳定性）
Q = np.diag([5.0, 5.0, 0.5])  # 减小状态权重
R = np.diag([0.5, 0.5])        # 增加输入权重
F = np.diag([10.0, 10.0, 1.0]) # 减小终端权重

# ========== 改进的线性化模型 ==========
def get_linearized_model(theta, v, dt=T):
    """改进的线性化模型，处理奇异点"""
    A = np.eye(3)
    B = np.zeros((3, 2))
    
    # 处理零速度情况
    if abs(v) < 1e-3:
        B[0, 0] = dt * np.cos(theta)
        B[1, 0] = dt * np.sin(theta)
        B[2, 1] = dt
    else:
        A[0, 2] = -dt * v * np.sin(theta)
        A[1, 2] = dt * v * np.cos(theta)
        B[0, 0] = dt * np.cos(theta)
        B[1, 0] = dt * np.sin(theta)
        B[2, 1] = dt
    
    return A, B

# ========== 改进的轨迹类 ==========
class Trajectory:
    def __init__(self, coeff, T):
        self.coeff = coeff
        self.T = T
        self.polyx = np.poly1d(self.coeff[0])
        self.polyy = np.poly1d(self.coeff[1])
        self.polyx_deriv = self.polyx.deriv()
        self.polyy_deriv = self.polyy.deriv()
        
        # 平滑处理参数
        self.alpha = 0.1  # 低通滤波系数
        self.prev_theta = 0.0
    
    def getPos(self, t):
        return [self.polyx(t), self.polyy(t)]
    
    def getVel(self, t):
        return [self.polyx_deriv(t), self.polyy_deriv(t)]
    
    def getTheta(self, t):
        """带平滑处理的朝向角计算"""
        vx, vy = self.getVel(t)
        speed = math.sqrt(vx**2 + vy**2)
        
        if speed < 1e-3:
            return self.prev_theta  # 使用上一次的值
        
        theta = math.atan2(vy, vx)
        
        # 角度平滑处理（避免突变）
        diff = theta - self.prev_theta
        if diff > math.pi:
            diff -= 2 * math.pi
        elif diff < -math.pi:
            diff += 2 * math.pi
            
        smooth_theta = self.prev_theta + self.alpha * diff
        self.prev_theta = smooth_theta
        
        return smooth_theta

# ========== 状态空间类 ==========
class StateSpace:
    def __init__(self):
        self.x = np.zeros((3, 1))  # [x, y, theta]
    
    def init(self, x):
        self.x = x.reshape(-1, 1)
    
    def step(self, u, dt=T):
        """带约束的差速驱动模型"""
        v, w = u
        theta = self.x[2, 0]
        
        # 应用速度约束
        v = np.clip(v, 0, MAX_SPEED)
        w = np.clip(w, -MAX_OMEGA, MAX_OMEGA)
        
        # 离散时间运动学
        self.x[0, 0] += v * dt * np.cos(theta)
        self.x[1, 0] += v * dt * np.sin(theta)
        self.x[2, 0] += w * dt
        
        # 角度归一化 [-π, π]
        self.x[2, 0] = (self.x[2, 0] + math.pi) % (2 * math.pi) - math.pi
        
        return self.x.copy()

# ========== 改进的MPC类 ==========
class MPC:
    def __init__(self, Q, R, F, N):
        self.Q = Q
        self.R = R
        self.F = F
        self.N = N
        self.d_x = 3
        self.d_u = 2
        self.fail_count = 0
    
    def step(self, x_ref, u_ref, x0, theta0, v0):
        """带错误处理的MPC求解"""
        # 获取线性化模型
        A, B = get_linearized_model(theta0, v0)
        
        # 定义优化变量
        x = cp.Variable((self.d_x, self.N + 1))
        u = cp.Variable((self.d_u, self.N))
        
        cost = 0
        constraints = []
        
        # 初始状态约束
        constraints.append(x[:, 0] == x0.flatten())
        
        # 构建代价函数和约束
        for t in range(self.N):
            # 状态误差代价
            state_error = x[:, t] - x_ref[:, t]
            cost += cp.quad_form(state_error, self.Q)
            
            # 输入误差代价
            input_error = u[:, t] - u_ref[:, t]
            cost += cp.quad_form(input_error, self.R)
            
            # 动力学约束（松弛处理）
            constraints.append(
                x[:, t + 1] == A @ x[:, t] + B @ u[:, t]
            )
            
            # 宽松的输入约束
            constraints.append(u[0, t] >= 0)
            constraints.append(u[0, t] <= MAX_SPEED * 1.5)  # 稍宽松
            constraints.append(cp.abs(u[1, t]) <= MAX_OMEGA * 1.5)
        
        # 终端代价
        terminal_error = x[:, self.N] - x_ref[:, self.N]
        cost += cp.quad_form(terminal_error, self.F)
        
        # 求解问题（带容错）
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            # 使用更鲁棒的求解器
            prob.solve(solver=cp.ECOS, max_iters=200, verbose=False)
            
            if prob.status == cp.OPTIMAL:
                self.fail_count = 0
                return u[:, 0].value, x.value
            else:
                self.fail_count += 1
                print(f"MPC求解失败 (状态: {prob.status})")
        except Exception as e:
            self.fail_count += 1
            print(f"求解器异常: {str(e)}")
        
        # 备用策略：使用参考输入或PD控制器
        if self.fail_count < 5:
            return u_ref[:, 0], None  # 使用参考输入
        else:
            # 简单的PD控制器作为后备
            kp = 0.8
            kd = 0.1
            pos_error = np.linalg.norm(x_ref[:2, 0] - x0[:2, 0])
            theta_error = x_ref[2, 0] - x0[2, 0]
            
            # 角度误差归一化
            theta_error = (theta_error + math.pi) % (2 * math.pi) - math.pi
            
            v = kp * pos_error
            w = kp * theta_error + kd * (theta_error - self.last_theta_error)
            self.last_theta_error = theta_error
            
            return np.array([v, w]), None

# ========== 主程序 ==========
# 创建更平滑的轨迹
px = [0.03, -0.01, 0.005, 0.02, -0.04]  # 减小系数
py = [0.01, 0.0, -0.02, 0.01, 0.05]
traj = Trajectory([px, py], 20)

# 初始化系统
ss = StateSpace()
pos = traj.getPos(0)
theta0 = traj.getTheta(0)
x_init = np.array([pos[0], pos[1], theta0])
ss.init(x_init)

# 创建MPC控制器
mpc = MPC(Q, R, F, N)
mpc.last_theta_error = 0.0  # 用于PD控制器

# 历史记录
t_hist = []
x_hist = []
y_hist = []
theta_hist = []
ref_x_hist = []
ref_y_hist = []
u_hist = []

# 主循环
for i in range(200):
    t = i * T
    t_hist.append(t)
    
    # 当前状态
    x0 = ss.x
    
    # 生成参考轨迹（更保守）
    x_ref = np.zeros((3, N + 1))
    u_ref = np.zeros((2, N))
    
    for j in range(N + 1):
        time_point = (i + j) * T
        pos = traj.getPos(time_point)
        theta_ref = traj.getTheta(time_point)
        x_ref[:, j] = [pos[0], pos[1], theta_ref]
        
        if j < N:
            vel = traj.getVel(time_point)
            v_ref = math.sqrt(vel[0]**2 + vel[1]**2)
            # 限制参考速度
            v_ref = min(v_ref, MAX_SPEED * 0.8)
            u_ref[:, j] = [v_ref, 0]
    
    # MPC求解
    u_opt, x_pred = mpc.step(
        x_ref, u_ref, 
        x0, x0[2, 0], u_ref[0, 0]  # 使用参考速度进行线性化
    )
    
    # 应用控制输入
    if u_opt is not None:
        ss.step(u_opt)
    
    # 记录数据
    x_hist.append(ss.x[0, 0])
    y_hist.append(ss.x[1, 0])
    theta_hist.append(ss.x[2, 0])
    
    ref_pos = traj.getPos(t)
    ref_x_hist.append(ref_pos[0])
    ref_y_hist.append(ref_pos[1])
    
    if u_opt is not None:
        u_hist.append(u_opt)
    else:
        u_hist.append([0, 0])

# ========== 可视化 ==========
plt.figure(figsize=(15, 10))

# 轨迹跟踪结果
plt.subplot(2, 2, 1)
plt.plot(x_hist, y_hist, 'b-', label='actual traj')
plt.plot(ref_x_hist, ref_y_hist, 'r--', label='ref traj')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('traj chasing')
plt.legend()
plt.axis('equal')

# 朝向角变化
plt.subplot(2, 2, 2)
plt.plot(t_hist, np.degrees(theta_hist), 'g-', label='actual orient')
plt.plot(t_hist, [np.degrees(traj.getTheta(t)) for t in t_hist], 'm--', label='ref orient')
plt.xlabel('time (s)')
plt.ylabel('angle (度)')
plt.title('orientation')
plt.legend()

# 控制输入
u_hist = np.array(u_hist)
plt.subplot(2, 2, 3)
plt.plot(t_hist, u_hist[:, 0], 'b-', label='speed (m/s)')
plt.axhline(y=MAX_SPEED, color='r', linestyle='--', label='speed limit')
plt.xlabel('time (s)')
plt.ylabel('speed')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t_hist, np.degrees(u_hist[:, 1]), 'r-', label='speed (deg/s)')
plt.axhline(y=np.degrees(MAX_OMEGA), color='g', linestyle='--', label='upper bound')
plt.axhline(y=-np.degrees(MAX_OMEGA), color='g', linestyle='--', label='lower bound')
plt.xlabel('time (s)')
plt.ylabel('speed')
plt.legend()

plt.tight_layout()
plt.show()
