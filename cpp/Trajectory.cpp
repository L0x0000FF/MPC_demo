#include "Trajectory.h"

Trajectory::Trajectory() {}

Trajectory::Trajectory(
    uint16_t N, std::vector<std::vector<std::vector<float>>> coeff_matrices,
    std::vector<float> durations, uint16_t row, uint16_t col) {
  this->coeff_matrices = coeff_matrices;
  this->N = N;
  this->durations = durations;
  this->row = row;
  this->col = col;
}

Vector3f Trajectory::get_pos(float t) {
  Vector3f result = {0.0f, 0.0f, 0.0f};
  uint16_t idx = 0;
  for (idx = 0; idx < N; idx++) {
    if (t < durations[idx]) {
      break;
    } else {
      t -= durations[idx];
    }
  }
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
  uint16_t idx = 0;
  for (idx = 0; idx < N; idx++) {
    if (t < durations[idx]) {
      break;
    } else {
      t -= durations[idx];
    }
  }
  float tn = 1.0;
  uint16_t n = 1;
  for (int16_t i = this->col - 1; i >= 0; i--) {
    result.x += n * tn * this->coeff_matrices[idx][0][i];
    result.y += n * tn * this->coeff_matrices[idx][1][i];
    result.z += n * tn * this->coeff_matrices[idx][2][i];
    tn *= t;
    n++;
  }
  return result;
}
