#include "Trajectory.h"
#include <iostream>

int main() {
  using namespace std;
  vector<float> p1a = {1, 2, 3, 4, 5};
  vector<float> p1b = {2, 3, 4, 5, 6};
  vector<float> p1c = {3, 4, 5, 6, 7};
  vector<vector<float>> coeff1 = {p1a, p1b, p1c};
  vector<float> p2a = {4, 6, 8, 7, 1};
  vector<float> p2b = {5, 1, 8, 9, 2};
  vector<float> p2c = {3, 6, 3, 4, 1};
  vector<vector<float>> coeff2 = {p2a, p2b, p2c};
  uint16_t N = 2;
  std::vector<float> durations = {1, 2};
  vector<vector<vector<float>>> coeff = {coeff1, coeff2};
  uint16_t row = 3;
  uint16_t col = 5;
  Trajectory a(N,coeff,durations,row,col);
  Trajectory b = a;
  Trajectory c(a);
  Trajectory* d = new Trajectory(a);
  Vector3f pos = a.get_pos(1.2);
  cout << pos.x << pos.y << pos.z << endl;
  delete d;
  return 0;
}