#pragma once

#include <vector>

#include <Eigen/Geometry>

#include "rtvm/pose.hpp"

namespace rtvm {

class PoseGraph {
public:
    // A measured relative motion between two keyframes.
    struct Edge {
        int from;
        int to;
        Pose measured;   // keyframe `to` seen from keyframe `from`
        bool loop;       // true: loop-closure edge (kept as measured by refreshChainEdges)
    };

    // Add current estimation of the position of the keyframe, returns the id.
    int addNode(const Pose& initial);

    // Add edge from keyframe `from` to keyframe `to`.
    // `measured` is the value the edge states (the relative motion from keyframe from to keyframe to that PnP measured).
    // `loop`: true for a loop-closure edge, false for a chain (odometry) edge.
    // Throws if the id `from` or `to` don't exist.
    void addEdge(int from, int to, const Pose& measured, bool loop = false);

    // Adjusts all poses to agree with all edges as well as possible; node 0 stays fixed.
    void optimize();

    // Returns the current pose of a node (optimized, once optimize() has run).
    // Throws if the id doesn't exist.
    Pose pose(int id) const;

    // Number of nodes and edges in the graph.
    int numNodes() const { return static_cast<int>(nodes_.size()); }
    int numEdges() const { return static_cast<int>(edges_.size()); }

    // Getters
    // The edge with this index (in the order added). Throws std::out_of_range for a bad index.
    const Edge& edge(int index) const;

    // Setters
    // Overwrite a node's pose (e.g. with the result of bundle adjustment). Throws for a bad id.
    void setPose(int id, const Pose& pose);

    // Re-measure every non-loop edge from the current poses: measured = relativePose(from, to).
    // After bundle adjustment moved the poses, the old PnP measurements would pull them back.
    // Loop edges keep their original measurement.
    void refreshChainEdges();

private:
    struct Node {
        Eigen::Vector3d p;
        Eigen::Quaterniond q;
    };
    void checkId(int id) const;
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
};

}  // namespace rtvm