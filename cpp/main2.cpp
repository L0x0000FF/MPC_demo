#include <iomanip>
#include <iostream>

#include "Trajectory.h"

int main() {
  using namespace std;
  // 测试数据：两段轨迹，每段是5阶多项式（3维：x,y,z）
  vector<float> p1a = {1, 2, 3, 4, 5};  // x系数: 1 + 2t + 3t² + 4t³ + 5t⁴
  vector<float> p1b = {2, 3, 4, 5, 6};  // y系数
  vector<float> p1c = {3, 4, 5, 6, 7};  // z系数
  vector<vector<float>> coeff1 = {p1a, p1b, p1c};

  vector<float> p2a = {4, 6, 8, 7, 1};  // 第二段x系数
  vector<float> p2b = {5, 1, 8, 9, 2};  // y系数
  vector<float> p2c = {3, 6, 3, 4, 1};  // z系数
  vector<vector<float>> coeff2 = {p2a, p2b, p2c};

  uint16_t N = 2;                         // 2段轨迹
  std::vector<float> durations = {1, 2};  // 第一段1秒，第二段2秒
  vector<vector<vector<float>>> coeff = {coeff1, coeff2};
  uint16_t row = 3;  // 3维（x,y,z）
  uint16_t col = 5;  // 5个系数（4阶多项式）

  // 1. 验证构造函数
  Trajectory a(N, coeff, durations, row, col);

  // 2. 验证拷贝构造
  Trajectory b = a;
  Trajectory c(a);
  Trajectory* d = new Trajectory(a);

  // 3. 验证get_pos()：分两段测试
  cout << fixed << setprecision(2);
  cout << "=== Position Tests ===" << endl;
  // 第一段轨迹 (t=0.5 < durations[0]=1)
  Vector3f pos1 = a.get_pos(0.5);
  cout << "t=0.5: x=" << pos1.x << ", y=" << pos1.y << ", z=" << pos1.z << endl;

  // 第二段轨迹 (t=1.2: 先减去第一段的1秒，剩余0.2秒)
  Vector3f pos2 = a.get_pos(1.2);
  cout << "t=1.2: x=" << pos2.x << ", y=" << pos2.y << ", z=" << pos2.z << endl;

  // 4. 验证get_vel()：速度是位置的多项式导数
  cout << "\n=== Velocity Tests ===" << endl;
  Vector3f vel1 = a.get_vel(0.5);
  cout << "t=0.5: vx=" << vel1.x << ", vy=" << vel1.y << ", vz=" << vel1.z
       << endl;

  Vector3f vel2 = a.get_vel(1.2);
  cout << "t=1.2: vx=" << vel2.x << ", vy=" << vel2.y << ", vz=" << vel2.z
       << endl;

  // 5. 边界测试 (t=3.0，超出总时长3秒)
  Vector3f pos3 = a.get_pos(3.0);
  cout << "\n=== Boundary Test (t=3.0) ===" << endl;
  cout << "x=" << pos3.x << ", y=" << pos3.y << ", z=" << pos3.z << endl;

  delete d;
  return 0;
}
