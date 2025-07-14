import numpy as np

class PID:
  def __init__(self,kp,ki,kd,output_limit):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.output = 0
    self.last_input = 0
    self.error = [0,0,0]
    self.output_limit = output_limit

  def calc(self,input,ref):
    e = ref - input
    self.error.pop(0)
    self.error.append(e)
    intergral = np.sum(self.error)
    self.output = self.kp * e + self.ki * intergral + self.kd * (self.error[2] - self.error[1]) 
    if self.output > self.output_limit:
      self.output = self.output_limit
    elif self.output < -self.output_limit:
      self.output = -self.output_limit
    return self.output
  def aaa(self):
    print("aaa")