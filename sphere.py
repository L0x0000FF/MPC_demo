import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from pid import PID
from trajectory import Trajectory

class Quaternion:
  def __init__(self,*args,**kwargs):
    if len(args) == 1:
      self.val = args[0]
    elif len(args) == 2:
      self.val = np.array([args[0],args[1][0],args[1][1],args[1][2]])
    elif len(args) == 4:
      self.val = np.array([args[0],args[1],args[2],args[3]])
    else:
      raise SyntaxError('Quaternion:wrong number of value')
  
  def __getitem__(self,i):
    return self.val[i]
  
  def __setitem__(self,i,val):
    self.val[i] = val

  def __matmul__(self,other):
    result = Quaternion(self[0]*other[0] - self[1]*other[1] - self[2]*other[2] - self[3]*other[3],
                        self[0]*other[1] + self[1]*other[0] + self[2]*other[3] - self[3]*other[2],
                        self[0]*other[2] + self[2]*other[0] + self[3]*other[1] - self[1]*other[3],
                        self[0]*other[3] + self[3]*other[0] + self[1]*other[2] - self[2]*other[1])
    return result
  
  def __add__(self,other):
    return Quaternion(self[0]+other[0],self[1]+other[1],self[2]+other[2],self[3]+other[3])
  
  def __sub__(self,other):
    return Quaternion(self[0]-other[0],self[1]-other[1],self[2]-other[2],self[3]-other[3])
  
  def __str__(self):
    return self.val.__str__()
  
  def __mul__(self,scalar:float):
    return Quaternion(self[0]*scalar,self[1]*scalar,self[2]*scalar,self[3]*scalar)
  
  def __truediv__(self,scalar:float):
    return Quaternion(self[0]/scalar,self[1]/scalar,self[2]/scalar,self[3]/scalar)
  
  def conj(self):
    return Quaternion(self[0],-self[1],-self[2],-self[3])
  
  def norm(self):
    return np.linalg.norm(self.val)
  
  def inv(self):
    return self.conj() / self.norm()**2
  
class model:
  def __init__(self,r):
    self.radius = r
    self.pos = np.array([r,0.0,0.0])
    self.orientation = np.array([1.0,0.0,0.0])
    self.v = 0
    self.w = 0
    self.q_vel = Quaternion(0,0,0,0)
  
  def initPos(self,pos:np.ndarray):
    self.pos = pos
  
  def initOrientation(self,orientation:np.ndarray):
    if not np.isclose(np.dot(orientation,self.pos),0,atol=1e-6):
      raise ValueError('initOrientaion: the orientation vector is not orthogonal to the position vector.')
    self.orientation = orientation

  def set_v_n_w(self,v,w):
    self.v = v
    self.w = w

  def step(self,dt):
    half_theta = self.w * dt / 2
    self.q_vel = Quaternion(np.cos(half_theta),np.sin(half_theta)*self.pos)
    q = self.q_vel @ Quaternion(0,self.orientation) @ self.q_vel.inv()
    self.orientation = np.array([q[1],q[2],q[3]])

    half_angle = self.v / self.radius / 2 * dt
    normal = np.cross(self.pos,self.orientation)
    normal /= np.linalg.norm(normal)
    q_forward = Quaternion(np.cos(half_angle),np.sin(half_angle)*normal)
    q = q_forward @ Quaternion(0,self.pos) @ q_forward.inv()
    self.pos = np.array([q[1],q[2],q[3]])

def draw_sphere(plot):
  # 创建球面网格
  u = np.linspace(0, 2 * np.pi, 100)  # 经度
  v = np.linspace(0, np.pi, 50)       # 纬度
  x = sp.radius * np.outer(np.cos(u), np.sin(v))
  y = sp.radius * np.outer(np.sin(u), np.sin(v))
  z = sp.radius * np.outer(np.ones(np.size(u)), np.cos(v))
  # 绘制球体表面
  plot.plot_surface(x, y, z, 
                  rstride=2, cstride=2, 
                  color='lightblue', 
                  alpha=0.3, 
                  edgecolor='grey', 
                  linewidth=0.5)
  
def draw_traj(plot,traj,color):
  plot.plot(traj[:,0],traj[:,1],traj[:,2],color)

sp = model(1.0)

figure = plt.figure()
ax = figure.add_subplot(111,projection='3d')
traj = Trajectory()
traj.initFromJson('./traj.json')
# traj
dt = 0.01
points = []
sp.initPos(traj.getPos(0))
sp.initOrientation(np.array([1,1,0]))
steering_pid = PID(1,0,0,20)
forward_pid = PID(10,0,0,10)
pos = []
e = []
for i in range(700):
  pt_ = traj.getPos(dt*i)
  pt = pt_ / np.linalg.norm(pt_) * sp.radius
  R_z = sp.pos / np.linalg.norm(sp.pos)
  R_x = sp.orientation / np.linalg.norm(sp.orientation)
  R_y = np.cross(R_z,R_x)
  R = np.column_stack([R_x,R_y,R_z])
  r = R.T @ (pt - sp.pos)
  theta = np.arccos((r[2]+sp.radius)/sp.radius)
  phi = np.arctan2(r[1],r[0])
  e.append(theta)
  w = steering_pid.calc(-phi,0)
  v = forward_pid.calc(-theta,0)
  # print(v,w)
  sp.set_v_n_w(v,w)
  sp.step(dt)
  pos.append(sp.pos)
  points.append(pt)

pos = np.asarray(pos)
points = np.asarray(points)
draw_traj(ax,pos,'b')
draw_traj(ax,points,'r')
ax.set_box_aspect([1,1,1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_zlim(-2,2)
f = plt.figure('e')
p = f.add_subplot()
p.plot(e)
plt.show()
