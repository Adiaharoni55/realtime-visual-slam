#include <random>

#include <ceres/ceres.h>
#include <gtest/gtest.h>

namespace {

// Residual for one point: the measured y minus the line's prediction m*x + c.
// It's a template because Ceres calls it with special number types that track
// derivatives (automatic differentiation), not only with plain doubles.
struct LineResidual {
    LineResidual(double x, double y) : x_(x), y_(y) {}

    template <typename T>
    bool operator()(const T* const m, const T* const c, T* residual) const {
        residual[0] = T(y_) - (m[0] * T(x_) + c[0]);
        return true;
    }

    double x_;
    double y_;
};

}  // namespace

// Toy problem to learn Ceres: fit y = m*x + c to noisy points on y = 2x + 1.
TEST(CeresTest, FitsLineToNoisyPoints) {
    const double true_m = 2.0;
    const double true_c = 1.0;
    std::mt19937 rng(42);                               // fixed seed: same data every run
    std::normal_distribution<double> noise(0.0, 0.1);   // noise with std 0.1

    // The unknowns, starting from a bad guess. Ceres changes them in place.
    double m = 0.0;
    double c = 0.0;

    ceres::Problem problem;
    for (int i = 0; i < 50; ++i) {
        const double x = 0.1 * i;
        const double y = true_m * x + true_c + noise(rng);
        // <functor, 1 residual, m has 1 value, c has 1 value>
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<LineResidual, 1, 1, 1>(new LineResidual(x, y)),
            nullptr,   // no robust loss (plain squared error)
            &m, &c);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;   // fine for tiny problems
    options.logging_type = ceres::SILENT;            // keep test output clean
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    ASSERT_TRUE(summary.IsSolutionUsable()) << summary.BriefReport();
    EXPECT_NEAR(m, true_m, 0.1);
    EXPECT_NEAR(c, true_c, 0.1);
}