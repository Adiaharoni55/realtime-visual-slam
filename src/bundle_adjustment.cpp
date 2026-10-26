#include "rtvm/bundle_adjustment.hpp"

#include <stdexcept>
#include <string>
#include <utility>

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

// ---- BundleAdjuster ----

void BundleAdjuster::checkPoseId(int id) const {
    if (id < 0 || id >= numPoses()) {
        throw std::out_of_range("BundleAdjuster: invalid pose id " + std::to_string(id));
    }
}

void BundleAdjuster::checkPointId(int id) const {
    if (id < 0 || id >= numPoints()) {
        throw std::out_of_range("BundleAdjuster: invalid point id " + std::to_string(id));
    }
}

BundleAdjuster::BundleAdjuster(PinholeCamera camera) : camera_(std::move(camera)) {}

int BundleAdjuster::addPose(const Pose& pose, bool fixed) {
    // Stored as position + quaternion: the form Ceres optimizes (as in the pose graph).
    poses_.push_back(PoseBlock{pose.translation(), Eigen::Quaterniond(pose.rotation()), fixed});
    return numPoses() - 1;
}

int BundleAdjuster::addPoint(const Eigen::Vector3d& world) {
    points_.push_back(world);
    return numPoints() - 1;
}

void BundleAdjuster::addObservation(int pose_id, int point_id, const Eigen::Vector2d& pixel) {
    checkPoseId(pose_id);
    checkPointId(point_id);
    observations_.push_back({pose_id, point_id, pixel});
}

void BundleAdjuster::optimize(double huber_delta) {
    // Nothing to optimize without observations (and Ceres can't handle an empty problem).
    if (observations_.empty()) {
        return;
    }
    ceres::Problem problem;

    // Shared by all blocks; the problem owns them and deletes each once.
    // The manifold keeps every quaternion a valid rotation (length 1), as in the pose graph.
    // Huber: a bad match far away pulls linearly, not quadratically, so it can't dominate.
    ceres::Manifold* quaternion_manifold = new ceres::EigenQuaternionManifold;
    ceres::LossFunction* loss = new ceres::HuberLoss(huber_delta);

    // One residual block per observation. Ceres works directly on poses_ and points_ (in place),
    // so nothing may be added to them while optimizing (push_back could move the memory).
    for (const Observation& o : observations_) {
        PoseBlock& cam = poses_[o.pose_id];
        Eigen::Vector3d& X = points_[o.point_id];
        problem.AddResidualBlock(ReprojectionError::create(o.pixel, camera_), loss,
                                 cam.p.data(), cam.q.coeffs().data(), X.data());
        problem.SetManifold(cam.q.coeffs().data(), quaternion_manifold);
    }

    // Fixed poses stay put. A pose without observations was never added to the problem,
    // and making an unknown block constant would abort, so check first.
    for (PoseBlock& pose : poses_) {
        if (pose.fixed && problem.HasParameterBlock(pose.p.data())) {
            problem.SetParameterBlockConstant(pose.p.data());
            problem.SetParameterBlockConstant(pose.q.coeffs().data());
        }
    }

    // BA structure: many points, each seen by a few poses. Schur eliminates the points first,
    // so the remaining system is only as large as the number of poses.
    ceres::Solver::Options options;
    options.linear_solver_type =
        options.sparse_linear_algebra_library_type != ceres::NO_SPARSE
            ? ceres::SPARSE_SCHUR
            : ceres::DENSE_SCHUR;
    options.logging_type = ceres::SILENT;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

Pose BundleAdjuster::pose(int id) const {
    checkPoseId(id);
    const PoseBlock& pb = poses_[id];
    Pose res = Pose::Identity();
    res.linear() = pb.q.normalized().toRotationMatrix();
    res.translation() = pb.p;
    return res;
}

Eigen::Vector3d BundleAdjuster::point(int id) const {
    checkPointId(id);
    return points_[id];
}

}  // namespace rtvm
