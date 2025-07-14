#include "Trajectory.h"
#include "stdio.h"

Trajectory::Trajectory() {}

Trajectory::Trajectory(
    uint16_t N, std::vector<std::vector<std::vector<float>>> coeff_matrices,
    std::vector<float> durations, uint16_t row, uint16_t col) {
  this->coeff_matrices = coeff_matrices;
  this->N = N;
  this->durations = durations;
  this->row = row;
  this->col = col;
  this->T_total = 0.0f;
  for(uint16_t i = 0;i < N;i++){
    this->T_total += durations[i];
  }
}

uint16_t Trajectory::get_idx(float& t) {
  uint16_t idx = 0;
  for (idx = 0; idx < N; idx++) {
    if (t < durations[idx]) {
      break;
    } else {
      t -= durations[idx];
    }
  }
  if(idx >= N)idx = N-1;
  return idx;
}

Vector3f Trajectory::get_pos(float t) {
  Vector3f result = {0.0f, 0.0f, 0.0f};
  uint16_t idx = get_idx(t);
  float tn = 1.0;
  for (int16_t i = this->col; i >= 0; i--) {
    result.x += tn * this->coeff_matrices[idx][0][i];
    result.y += tn * this->coeff_matrices[idx][1][i];
    result.z += tn * this->coeff_matrices[idx][2][i];
    tn *= t;
  }
  return result;
}

Vector3f Trajectory::get_vel(float t) {
  Vector3f result = {0.0f, 0.0f, 0.0f};
  uint16_t idx = get_idx(t);
  float tn = 1.0;
  uint16_t n = 1;
  for (int16_t i = this->col-1; i >= 0; i--) {
    result.x += n * tn * this->coeff_matrices[idx][0][i];
    result.y += n * tn * this->coeff_matrices[idx][1][i];
    result.z += n * tn * this->coeff_matrices[idx][2][i];
    tn *= t;
    n++;
  }
  return result;
}
