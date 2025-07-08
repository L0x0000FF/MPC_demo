import numpy as np
import json

class Piece:
  def __init__(self,coeff,duration):
    self.coeff = coeff
    self.duration = duration
    self.polyx = np.poly1d(self.coeff[0])
    self.polyy = np.poly1d(self.coeff[1])
    self.polyz = np.poly1d(self.coeff[2])
    self.polyvx = np.poly1d(np.polyder(self.coeff[0]))
    self.polyvy = np.poly1d(np.polyder(self.coeff[1]))
    self.polyvz = np.poly1d(np.polyder(self.coeff[2]))
  
  def getPos(self,t):
    return np.array([self.polyx(t),self.polyy(t),self.polyz(t)])

  def getVel(self,t):
    return np.array([self.polyvx(t),self.polyvy(t),self.polyvz(t)])
  
class Trajectory:
  def __init__(self,durations=None,pieces=None):
    self.pieces = pieces if pieces is not None else []
    self.durations = durations if durations is not None else 0
    self.N = 0
  
  def initFromJson(self,filename):
    with open(filename, 'r') as f:
      matrix_json = json.load(f)
    self.N = matrix_json['N']
    self.durations = matrix_json['durations']
    coeff = matrix_json['coeff']
    for i in range(self.N):
      self.pieces.append(Piece(coeff[i],self.durations[i]))

  def getIdx(self,t:float):
    idx = 0
    while idx < self.N:
      if t > self.durations[idx]:
        t  -= self.durations[idx]
        idx += 1
      else:
        break
    return idx,t

  def getPos(self,t:float):
    idx,t_res = self.getIdx(t)
    return self.pieces[idx].getPos(t_res)
  
  def getVel(self,t:float):
    idx,t_res = self.getIdx(t)
    return self.pieces[idx].getVel(t_res)
  
if __name__ == '__main__':
  traj = Trajectory()
  traj.initFromJson('./traj.json')
  for i in range(60):
    t = i * 0.1
    print(traj.getVel(t))
