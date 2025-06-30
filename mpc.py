import numpy as np
import cvxpy
import scipy
import scipy.linalg
from matplotlib import pyplot as plt

N = 6

A = [
  [0, 1, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 1],
  [0, 0, 0, 0]
]

B = [
  [0, 0],
  [1, 0],
  [0, 0],
  [0, 1]
]

A = np.array(A)
B = np.array(B)

Q = np.array([
  [100,0,0,0],
  [0,0.01,0,0],
  [0,0,100,0],
  [0,0,0,0.01]
])
R = np.array([
  [0.01,0],
  [0,0.01]
])
F = np.array([
  [1,0,0,0],
  [0,1,0,0],
  [0,0,1,0],
  [0,0,0,1]
])

class Trajectory:
  def __init__(self,coeff,T):
    self.coeff = coeff
    self.T = T
    self.polyx = np.poly1d(self.coeff[0])
    self.polyy = np.poly1d(self.coeff[1])
  
  def getPos(self,t):
    return [self.polyx(t),self.polyy(t)]

  def getVel(self,t,t_prev):
    return [(self.polyx(t)-self.polyx(t_prev))/(t-t_prev),(self.polyy(t)-self.polyy(t_prev))/(t-t_prev)]

class StateSpace:
  def __init__(self, A:np.ndarray, B:np.ndarray, C=None, D=None):
    self.A = A
    self.B = B
    self.d_x = A.shape[1]
    self.d_u = B.shape[1]
    self.C = C if C is not None else np.identity(self.d_x)
    self.d_y = self.C.shape[0]
    self.D = D if D is not None else np.zeros((self.d_y,self.d_u))
    self.x = np.zeros((self.d_x,1))
    self.y = np.zeros((self.d_y,1))
  
  def init(self, x:np.ndarray):
    self.x = x

  def step(self, u:np.ndarray):
    self.x = self.A @ self.x + self.B @ u
    self.y = self.C @ self.x + self.D @ u
    # print("---------")
    # print(self.x,self.y)
    return self.y

class MPC:
  def __init__(self,A:np.ndarray, B:np.ndarray, Q:np.ndarray, R:np.ndarray, F:np.ndarray, N:int):
    self.A = A
    self.B = B
    self.Q = Q
    self.R = R
    self.F = F
    self.N = N
    self.d_x = A.shape[1]
    self.x = None
    self.d_u = B.shape[1]
    self.u = None
    
  def step(self,x_ref:np.ndarray,x0:np.ndarray):
    self.x = cvxpy.Variable((self.d_x,self.N+1))
    self.u = cvxpy.Variable((self.d_u,self.N))
    J = 0
    constraints = []
    for i in range(N):
      J += cvxpy.quad_form(self.u[:,i], self.R)
      if i > 0:
        J += cvxpy.quad_form(self.x[:,i] - x_ref[:,i], self.Q)
      constraints.append(self.x[:,i+1] - x_ref[:,i+1] == self.A @ (self.x[:,i] - x_ref[:,i]) + self.B @ self.u[:,i])
    J += cvxpy.quad_form(self.x[:,N] - x_ref[:,N], self.F)
    constraints.append(self.x[:,0] == x0.squeeze(1))
    prob = cvxpy.Problem(cvxpy.Minimize(J),constraints)
    prob.solve(solver=cvxpy.ECOS,verbose=False)
    return self.u.value[:,0]

T = 0.01
px = [3,-1,0.5,2,-4]
py = [1,0,-2,1,5]
traj = Trajectory([px,py],20)
ss = StateSpace(A*T + np.identity(4),B)
pos = traj.getPos(0)
x_init = np.array([pos[0],0,pos[1],0])
x_init = x_init.reshape(-1,1)
ss.init(x_init)
mpc = MPC(A*T + np.identity(4),B,Q,R,F,N)
t = []
x = []
y = []
ref_x = []
ref_y = []
u_all = []
for i in range(100):
  dt = i * T
  t.append(dt)
  x0 = ss.x
  # x_ref = [x0.transpose().squeeze(0)]
  x_ref = []
  for j in range(N+1):
    pos = traj.getPos((i+j)*T)
    vel = traj.getVel((i+j)*T,(i+j-1)*T)
    x_ref.append(np.array([pos[0],vel[0],pos[1],vel[1]]))
  x_ref = np.array(x_ref)
  x_ref = np.array(x_ref).transpose()
  u = mpc.step(x_ref,x0)
  u = u.reshape(-1,1)
  output = ss.step(u)
  x.append(output[0])
  y.append(output[2])
  ref = traj.getPos(dt)
  ref_x.append(ref[0])
  ref_y.append(ref[1])
  u_all.append(u)

t = np.array(t)
x = np.array(x)
y = np.array(y)
print(x)
print(y)
plt.figure('x')
plt.plot(x,y,color='b')
plt.plot(ref_x,ref_y,color='r')
plt.figure('u')
u_all = np.array(u_all)
plt.subplot(1,2,1)
plt.plot(u_all[0,:])
plt.subplot(1,2,2)
plt.plot(u_all[1,:])
plt.show()
