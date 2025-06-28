import numpy as np
import cvxpy
import scipy
import scipy.linalg
from matplotlib import pyplot as plt

N = 4

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

Q = np.identity(4)
R = np.identity(2)
F = np.identity(4)

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
    
    # self.Q_hat = None
    # self.R_hat = None
    # for i in range(self.N):
    #   if self.Q_hat == None:
    #     self.Q_hat = self.Q
    #   else:
    #     self.Q_hat = scipy.linalg.block_diag(self.Q_hat,self.Q)
    # self.Q_hat = scipy.linalg.block_diag(self.Q_hat,self.F)

    # for i in range(self.N):
    #   if self.R_hat == None:
    #     self.R_hat = self.R
    #   else:
    #     self.R_hat = scipy.linalg.block_diag(self.R_hat,self.R)
    # # get M&C
    # self.M = np.identity(self.d_x)
    # tmp = self.A
    # for i in range(N):
    #   self.M = np.concatenate((self.M,tmp),axis=0)
    #   tmp = tmp @ self.A
    
  def step(self,x_ref:np.ndarray,x0:np.ndarray):
    self.x = cvxpy.Variable((self.d_x,self.N+1))
    self.u = cvxpy.Variable((self.d_u,self.N))
    J = 0
    constraints = []
    for i in range(N):
      J += cvxpy.quad_form(self.u[:,i], self.R)
      if i > 0:
        J += cvxpy.quad_form(self.x[:,i] - x_ref[:,i], self.Q)
      constraints.append(self.x[:i+1] - x_ref[:,i+1] == self.A @ (self.x[:,i] - x_ref[:,i]) + self.B @ self.u[:,i])
    J += cvxpy.quad_form(self.x[:,N] - x_ref[:,N], self.F)
    constraints.append(self.x[:,0] == x0)
    prob = cvxpy.Problem(cvxpy.Minimize(J),constraints)
    prob.solve(solver=cvxpy.ECOS,verbose=False)
    return self.u[:,0]

ss = StateSpace(A,B)
ss.init(np.zeros((4,1)))
t = []
x = []
y = []
for i in range(100):
  dt = i / 10
  t.append(dt)
  u = np.array([[0.2*np.cos(dt)],[0.3*np.sin(dt)]])
  output = ss.step(u)
  x.append(output[0,0])
  y.append(output[2,0])

t = np.array(t)
x = np.array(x)
y = np.array(y)
plt.plot(y,x)
plt.show()