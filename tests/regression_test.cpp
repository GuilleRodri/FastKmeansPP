#include <iostream>
#include <cmath>
#include <string>
#include <iomanip>
#include "golden_generated.hpp"
#include "../fastkmeanspp.hpp"

static bool approx_equal(double a, double b, double rtol = 1e-9, double atol = 1e-9) {
  double diff = std::fabs(a - b);
  if (diff <= atol) return true;
  double denom = std::max(std::fabs(a), std::fabs(b));
  return diff <= rtol * (denom > 0 ? denom : 1.0);
}

static const char* acc_name(int acc) {
  if (acc == 0) return "Std";
  if (acc == 1) return "TIE";
  return "TIE+Norm";
}

static long double fingerprint(const FastKMeansPP::Matrix& C) {
  long double fp = 0.0L;
  for (int i = 0; i < C.rows(); ++i) {
    for (int j = 0; j < C.cols(); ++j) {
      fp += (long double)(i + 1) * (long double)(j + 1) * (long double)C(i, j);
    }
  }
  return fp;
}

int main() {
  std::cout << std::setprecision(17) << std::fixed;

  int failures = 0;

  for (std::size_t ci = 0; ci < GOLDEN_CASES_COUNT; ++ci) {
    const auto& tc = GOLDEN_CASES[ci];

    for (int acc = 0; acc < 3; ++acc) {
      FastKMeansPP::Matrix centers;
      double score = FastKMeansPP::run_kmeanspp(std::string(tc.file), centers, tc.k, tc.seed, acc);
      double fp = (double)fingerprint(centers);

      bool ok_score = approx_equal(score, tc.score[acc]);
      bool ok_fp    = approx_equal(fp, tc.fp[acc]);

      if (!ok_score || !ok_fp) {
        failures++;
        std::cerr << "[FAIL] " << tc.name << " (" << tc.file << "), k=" << tc.k
                  << ", seed=" << tc.seed << ", acc=" << acc_name(acc) << "\n";
        if (!ok_score) {
          std::cerr << "  score expected=" << tc.score[acc] << " got=" << score << "\n";
        }
        if (!ok_fp) {
          std::cerr << "  fp    expected=" << tc.fp[acc] << " got=" << fp << "\n";
        }
      }
    }
  }

  if (failures == 0) {
    std::cout << "[OK] Regression tests passed (" << (GOLDEN_CASES_COUNT * 3) << " checks).\n";
    return 0;
  } else {
    std::cerr << "[ERROR] " << failures << " checks failed.\n";
    std::cerr << "If changes are intentional, run: make update (and commit new golden files).\n";
    return 1;
  }
}

