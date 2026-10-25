#pragma once

#include <vector>

#include <Eigen/Geometry>

#include "rtvm/pose.hpp"

namespace rtvm {

class PoseGraph {
public:
    // Add current estimation of the position of the keyframe, returns the id.
    int addNode(const Pose& initial);

    // Add edge from keyframe `from` to keyframe `to`.
    // `measured` is the value the edge states (the relative motion from keyframe from to keyframe to that PnP measured).
    // Throws if the id `from` or `to` don't exist.
    void addEdge(int from, int to, const Pose& measured);

    // Adjusts all poses to agree with all edges as well as possible; node 0 stays fixed.
    void optimize();

    // Returns the current pose of a node (optimized, once optimize() has run). 
    // Throws if the id doesn't exist.
    Pose pose(int id) const;

    // Number of nodes in the graph.
    int numNodes() const { return static_cast<int>(nodes_.size()); }

private:
    struct Node { Eigen::Vector3d p; Eigen::Quaterniond q; };
    struct Edge { int from; int to; Pose measured; };
    void checkId(int id) const;

    std::vector<Node> nodes_;
    std::vector<Edge> edges_;

};

}  // namespace rtvm