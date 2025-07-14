#ifndef TRAJECTORY_H
#define TRAJECTORY_H

#include <vector>

#include "stdint.h"

typedef struct _Vector3f {
  float x;
  float y;
  float z;
} Vector3f;

class Trajectory {
 private:
  uint16_t N;  // number of pieces
  std::vector<std::vector<std::vector<float>>> coeff_matrices;
  std::vector<float> durations;
  uint16_t row;
  uint16_t col;

 public:
  float T_total;
  Trajectory();
  Trajectory(uint16_t N,
             std::vector<std::vector<std::vector<float>>> coeff_matrices,
             std::vector<float> durations, uint16_t row, uint16_t col);
  uint16_t get_idx(float& t);
  Vector3f get_pos(float t);
  Vector3f get_vel(float t);
};

#endif
