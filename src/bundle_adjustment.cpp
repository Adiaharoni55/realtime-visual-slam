#include "rtvm/bundle_adjustment.hpp"

#include <stdexcept>

#include <ceres/ceres.h>
#include <Eigen/Geometry>

namespace rtvm {

namespace {

// Residual for one observation: projected minus observed pixel (2 numbers).
struct ReprojectionError {
    ReprojectionError(const Eigen::Vector2d& observed, const PinholeCamera& camera)
        : u_obs_(observed.x()), v_obs_(observed.y()),
          fx_(camera.fx), fy_(camera.fy), cx_(camera.cx), cy_(camera.cy) {}

    template <typename T>
    bool operator()(const T* p_ptr, const T* q_ptr, const T* X_ptr, T* r_ptr) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p(p_ptr);     // camera position: 3 numbers
        Eigen::Map<const Eigen::Quaternion<T>>   q(q_ptr);     // camera rotation: 4 numbers
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> X(X_ptr);     // the 3D point: 3 numbers

        // The pose is camera -> world, so invert it: world point -> camera frame.
        const Eigen::Matrix<T, 3, 1> X_cam = q.conjugate() * (X - p);
        // Behind (or at) the camera: the division would explode. Tell Ceres this step is invalid.
        if (X_cam.z() < T(1e-6)) return false;

        // Pinhole projection (no distortion), then projected - observed.
        const T u = T(fx_) * X_cam.x() / X_cam.z() + T(cx_);
        const T v = T(fy_) * X_cam.y() / X_cam.z() + T(cy_);
        r_ptr[0] = u - T(u_obs_);
        r_ptr[1] = v - T(v_obs_);
        return true;
    }

    static ceres::CostFunction* create(const Eigen::Vector2d& observed, const PinholeCamera& camera) {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 4, 3>(
            new ReprojectionError(observed, camera));
    }

    double u_obs_, v_obs_, fx_, fy_, cx_, cy_;
};

}  // namespace

// The public function from the header. It uses the struct above with plain doubles.
Eigen::Vector2d reprojectionResidual(const PinholeCamera& camera, const Pose& camera_pose,
                                     const Eigen::Vector3d& world_point,
                                     const Eigen::Vector2d& observed_pixel) {
    const Eigen::Vector3d p = camera_pose.translation();
    const Eigen::Quaterniond q(camera_pose.rotation());   // coeffs() order: x, y, z, w, as Map expects
    const ReprojectionError error(observed_pixel, camera);
    Eigen::Vector2d residual;
    if (!error(p.data(), q.coeffs().data(), world_point.data(), residual.data())) {
        throw std::domain_error("reprojectionResidual: point is behind the camera");
    }
    return residual;
}

}  // namespace rtvm
