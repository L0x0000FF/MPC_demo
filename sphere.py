import numpy as np
from matplotlib import pyplot as plt

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
    self.orientation = orientation

  def set_v_n_w(self,v,w):
    self.v = v
    self.w = w

  def step(self,dt):
    self.pos += self.v * self.orientation * dt
    self.pos = self.pos / np.linalg.norm(self.pos) * self.radius
    pos_ = self.pos / np.linalg.norm(self.pos)
    half_theta = self.w * dt / 2
    self.q_vel = Quaternion(np.cos(half_theta),np.sin(half_theta)*pos_)
    q = self.q_vel @ Quaternion(0,self.orientation) @ self.q_vel.inv()
    self.orientation = np.array([q[1],q[2],q[3]])

pos = []
sp = model(1.0)
sp.initOrientation(np.array([0,0,1]))
for i in range(50):
  sp.set_v_n_w(0.1,0.5)
  sp.step(0.1)
  pos.append(sp.pos)

pos = np.asarray(pos)
print(pos)
figure = plt.figure()
ax = figure.add_subplot(111,projection='3d')
ax.plot(pos[:,0],pos[:,1],pos[:,2])
ax.set_box_aspect([1,1,1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_zlim(-2,2)
plt.show()
