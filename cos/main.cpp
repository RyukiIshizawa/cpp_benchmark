#include <bits/stdc++.h>

#include <eigen3/Eigen/Dense>

#define DIMENSION 2

using Vec = Eigen::Vector2d;
using fitness_type = double;

/// Weierstrass関数の初期化フラグ
bool FWeierstrass_init;
/// Weierstrass関数のk_max
constexpr int32_t FWeierstrass_k_max = 20;
/// Weierstrass関数のsum2
fitness_type FWeierstrass_sum2;
/// Weierstrass関数の累乗係数
std::vector<fitness_type> FWeierstrass_a_pow, FWeierstrass_bb_pow;
Eigen::Vector<fitness_type, FWeierstrass_k_max + 1> FWeierstrass_a_pow_mat,
    FWeierstrass_bb_pow_mat;

void FWeierstrass(const std::vector<Vec>& x, std::vector<fitness_type>& y,
                  int32_t l, int32_t n) {
  if (!FWeierstrass_init) {
    fitness_type a(0.5), b(3.0);
    FWeierstrass_a_pow = std::vector<fitness_type>(FWeierstrass_k_max + 1);
    FWeierstrass_bb_pow = std::vector<fitness_type>(FWeierstrass_k_max + 1);
    FWeierstrass_init = true;
    FWeierstrass_a_pow[0] = 1.0;
    FWeierstrass_bb_pow[0] = 2.0 * std::numbers::pi;
    FWeierstrass_a_pow_mat[0] = 1.0;
    FWeierstrass_bb_pow_mat[0] = 2.0 * std::numbers::pi;
    for (int32_t i = 1; i <= FWeierstrass_k_max; i++) {
      FWeierstrass_a_pow[i] = FWeierstrass_a_pow[i - 1] * a;
      FWeierstrass_bb_pow[i] = FWeierstrass_bb_pow[i - 1] * b;

      FWeierstrass_a_pow_mat[i] = FWeierstrass_a_pow[i];
      FWeierstrass_bb_pow_mat[i] = FWeierstrass_bb_pow[i];
    }
    for (int32_t j = 0; j <= FWeierstrass_k_max; ++j) {
      FWeierstrass_sum2 +=
          FWeierstrass_a_pow[j] * cos(FWeierstrass_bb_pow[j] * (0.5));
    }
    FWeierstrass_sum2 *= (double)x[0].rows();
  }

#if DIMENSION == 2
  // for (int i = l; i < n + l; i++) {
  //   y[i] = 0;
  //   for (int32_t k = 0; k <= FWeierstrass_k_max; ++k) {
  //     y[i] += FWeierstrass_a_pow[k] *
  //             (cos(FWeierstrass_bb_pow[k] * (x[i][0] + 0.5)) +
  //              cos(FWeierstrass_bb_pow[k] * (x[i][1] + 0.5)));
  //   }
  //   y[i] -= FWeierstrass_sum2;
  // }
  for (int i = l; i < n + l; i++) {
    Eigen::Matrix<fitness_type, DIMENSION, FWeierstrass_k_max + 1> tmp;
    tmp.colwise() = x[i];
    tmp.array() += 0.5;
    tmp.array().rowwise() *= FWeierstrass_bb_pow_mat.transpose().array();
    tmp = tmp.array().cos();
    y[i] = (tmp.colwise().sum() * FWeierstrass_a_pow_mat).sum() -
           FWeierstrass_sum2;
  }

#else
  double tmp;
  for (int32_t i = l; i < n + l; i++) {
    y[i] = -FWeierstrass_sum2;
    for (int32_t j = 0; j <= FWeierstrass_k_max; ++j) {
      tmp = 0;
      for (int32_t k = 0; k < x[0].rows(); k++) {
        tmp += cos(FWeierstrass_bb_pow[j] * (x[i][k] + 0.5));
      }
      y[i] += FWeierstrass_a_pow[j] * tmp;
    }
  }
#endif
}

int main() {
  std::vector<Vec> x(20);
  std::vector<fitness_type> y(20);
  for (int i = 0; i < (int)x.size(); i++) {
    x[i].fill(i*1.1 + 0.1);
  }
  for (int i = 0; i < 1000000; i++) FWeierstrass(x, y, 0, 20);
  for (int i = 0; i < (int)y.size(); i++) {
    std::cout << y[i] << std::endl;
  }

  return 0;
}