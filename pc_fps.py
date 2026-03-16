#!/home/hxd/miniconda3/envs/ros_pc/bin/python
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def farthest_point_sample(xyz, n_samples, farthest):
    """Numba FPS: 减少中间数组分配，提升吞吐。"""
    n_pts = xyz.shape[0]
    centroids = np.empty(n_samples, dtype=np.int32)
    distance = np.full(n_pts, 1e10, dtype=np.float32)

    for i in range(n_samples):
        centroids[i] = farthest

        cx = xyz[farthest, 0]
        cy = xyz[farthest, 1]
        cz = xyz[farthest, 2]

        for j in range(n_pts):
            dx = xyz[j, 0] - cx
            dy = xyz[j, 1] - cy
            dz = xyz[j, 2] - cz
            dist = dx * dx + dy * dy + dz * dz
            if dist < distance[j]:
                distance[j] = dist

        max_dist = -1.0
        max_idx = 0
        for j in range(n_pts):
            if distance[j] > max_dist:
                max_dist = distance[j]
                max_idx = j
        farthest = max_idx

    return centroids


class PointCloudFilter(Node):
    def __init__(self):
        super().__init__('PY_pointcloud_filter_node')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub = self.create_subscription(
            PointCloud2,
            '/camera/depth/color/points',
            self.callback,
            qos_profile
        )

        self.pub_sampled = self.create_publisher(
            PointCloud2,
            '/camera/sampled_points',
            10
        )

        # 优先保证频率的参数
        self.target_points = 1024
        self.prefilter_points = 2048
        self.use_fps = True
        self.rng = np.random.default_rng()

        self.work_space = np.array([
            [-0.5, 0.6],
            [-0.35, 0.4],
            [0.0, 1.3]
        ], dtype=np.float32)
        self.x_min, self.x_max = self.work_space[0]
        self.y_min, self.y_max = self.work_space[1]
        self.z_min, self.z_max = self.work_space[2]

        self.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
        ]

        # 预热 Numba，避免首帧突发延迟
        dummy = np.zeros((self.target_points, 3), dtype=np.float32)
        _ = farthest_point_sample(dummy, min(16, self.target_points), 0)

        self._stat_last_t = self.get_clock().now()
        self._stat_frames = 0
        self.get_logger().info('点云采样节点已启动 (ROS2, optimized)')

    def callback(self, msg):
        try:
            # 1) 直接读成 Nx4 float32，避免 list(generator) 的 Python 开销
            points_xyzrgb = pc2.read_points_numpy(
                msg,
                field_names=('x', 'y', 'z', 'rgb'),
                skip_nans=True
            ).astype(np.float32, copy=False)

            if points_xyzrgb.ndim != 2 or points_xyzrgb.shape[1] != 4:
                return
            if points_xyzrgb.shape[0] == 0:
                return

            # 2) 空间剪裁
            mask = (
                (points_xyzrgb[:, 0] > self.x_min) &
                (points_xyzrgb[:, 0] < self.x_max) &
                (points_xyzrgb[:, 1] > self.y_min) &
                (points_xyzrgb[:, 1] < self.y_max) &
                (points_xyzrgb[:, 2] > self.z_min) &
                (points_xyzrgb[:, 2] < self.z_max)
            )
            filtered = points_xyzrgb[mask]

            n_filtered = filtered.shape[0]
            if n_filtered == 0:
                return

            # 3) 先粗采样，限制 FPS 输入规模
            if n_filtered > self.prefilter_points:
                idx = self.rng.choice(n_filtered, self.prefilter_points, replace=False)
                filtered = filtered[idx]
                n_filtered = self.prefilter_points

            # 4) 采样到固定目标点数
            if n_filtered >= self.target_points:
                if self.use_fps:
                    seed = int(self.rng.integers(0, n_filtered))
                    idx_fps = farthest_point_sample(filtered[:, :3], self.target_points, seed)
                    sampled = filtered[idx_fps]
                else:
                    idx = self.rng.choice(n_filtered, self.target_points, replace=False)
                    sampled = filtered[idx]
            else:
                idx = self.rng.integers(0, n_filtered, size=self.target_points)
                sampled = filtered[idx]

            # 5) 手工构造 PointCloud2，减少 create_cloud 的 Python 负担
            out_msg = PointCloud2()
            out_msg.header = msg.header
            out_msg.height = 1
            out_msg.width = self.target_points
            out_msg.fields = self.fields
            out_msg.is_bigendian = False
            out_msg.point_step = 16
            out_msg.row_step = out_msg.point_step * out_msg.width
            out_msg.is_dense = True
            out_msg.data = sampled.astype(np.float32, copy=False).tobytes()
            self.pub_sampled.publish(out_msg)

            self._stat_frames += 1
            now = self.get_clock().now()
            dt_ns = (now - self._stat_last_t).nanoseconds
            if dt_ns > 5_000_000_000:
                hz = self._stat_frames / (dt_ns / 1e9)
                self.get_logger().info(f'pc_fps publish rate: {hz:.1f} Hz')
                self._stat_frames = 0
                self._stat_last_t = now

        except Exception as e:
            self.get_logger().error(f'处理异常: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
