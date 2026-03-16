import time
from pathlib import Path
from collections import deque
import numpy as np

import torch
from omegaconf import OmegaConf

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import Pose, PoseStamped  # 新增：用于发布位姿IK

# 导入你指定的自定义接口
from my_interfaces.msg import HeaderFloat32

# 导入 DP3 的环境
from train import TrainDP3Workspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

class DP3InferenceNodeIK(Node):
    def __init__(self, cfg, policy, device, n_obs_steps):
        super().__init__('dp3_inference_node_ik')
        self.cfg = cfg
        self.policy = policy
        self.device = device
        self.n_obs_steps = n_obs_steps
        
        # --- 参数配置 ---
        self.control_rate = 20.0  
        self.use_pc_color = bool(OmegaConf.select(cfg, "policy.use_pc_color", default=False))
        self.n_action_steps = int(OmegaConf.select(cfg, "n_action_steps", default=1))
        # 默认action shape如果配置没写明，这里假设改为8维
        action_shape = OmegaConf.select(cfg, "shape_meta.action.shape", default=[8])
        self.action_dim = int(action_shape[0]) if len(action_shape) > 0 else 8
        pc_shape = OmegaConf.select(cfg, "shape_meta.obs.point_cloud.shape", default=[1024, 3])
        self.pc_num_points = int(pc_shape[0]) if len(pc_shape) > 0 else 1024
        self.pc_feature_dim = int(pc_shape[1]) if len(pc_shape) > 1 else (6 if self.use_pc_color else 3)

        self.current_action_chunk = None
        self.current_action_idx = 0
        self.collecting_obs = True
        self.last_wait_log_time = 0.0
        self.last_run_log_time = 0.0
        self.obs_state_window = deque(maxlen=self.n_obs_steps)
        self.obs_pc_window = deque(maxlen=self.n_obs_steps)
        self.infer_success_count = 0
        self.infer_fail_count = 0
        self.published_action_count = 0

        # --- 使用 ROS2 近似时间同步订阅观测 ---
        self.sub_arm_pose = Subscriber(self, Pose, '/right_arm/rm_driver/udp_arm_position')
        self.sub_pointcloud = Subscriber(self, PointCloud2, 'camera/sampled_points')
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_arm_pose, self.sub_pointcloud],
            queue_size=max(20, self.n_obs_steps * 2),
            slop=0.05,
            allow_headerless=True
        )
        self.sync.registerCallback(self.synced_obs_cb)
        
        # --- 发布者 ---
        # 修改为发布 PoseStamped 到 right_arm/ik_target_pose
        self.pub_arm_cmd = self.create_publisher(PoseStamped, '/right_arm/ik_target_pose', 10)
        self.pub_gripper_cmd = self.create_publisher(HeaderFloat32, 'right_arm/gripper_cmd', 10)
        
        # --- 定时器 ---
        self.timer = self.create_timer(1.0 / self.control_rate, self.control_loop)
        
        self.get_logger().info(f"DP3 Node Started. Control: {self.control_rate}Hz, Actions: {self.n_action_steps} steps.")

    def _pointcloud_to_array(self, msg: PointCloud2) -> np.ndarray:
        field_names = [f.name for f in msg.fields]
        has_separate_rgb = all(name in field_names for name in ("r", "g", "b"))
        has_packed_rgb = ("rgb" in field_names) or ("rgba" in field_names)

        if self.use_pc_color and (has_separate_rgb or has_packed_rgb):
            if has_separate_rgb:
                points_gen = pc2.read_points(msg, field_names=("x", "y", "z", "r", "g", "b"), skip_nans=True)
                pc_array = self._points_to_float_array(points_gen, ("x", "y", "z", "r", "g", "b"))
                if pc_array.shape[1] >= 6 and np.max(pc_array[:, 3:6]) > 1.0:
                    pc_array[:, 3:6] = pc_array[:, 3:6] / 255.0
            else:
                packed_name = "rgb" if "rgb" in field_names else "rgba"
                points_gen = pc2.read_points(msg, field_names=("x", "y", "z", packed_name), skip_nans=True)
                raw_array = self._points_to_float_array(points_gen, ("x", "y", "z", packed_name))
                if len(raw_array) == 0:
                    return None
                xyz = raw_array[:, :3].astype(np.float32)
                rgb = np.stack([self._decode_packed_rgb(v) for v in raw_array[:, 3]], axis=0)
                pc_array = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
        else:
            points_gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            xyz_array = self._points_to_float_array(points_gen, ("x", "y", "z"))
            if len(xyz_array) == 0:
                return None
            if self.use_pc_color:
                zero_color = np.zeros((xyz_array.shape[0], 3), dtype=np.float32)
                pc_array = np.concatenate([xyz_array, zero_color], axis=1)
            else:
                pc_array = xyz_array
        return self._adapt_point_cloud_shape(pc_array)

    @staticmethod
    def _decode_packed_rgb(rgb_value) -> np.ndarray:
        packed = np.array([rgb_value], dtype=np.float32).view(np.uint32)[0]
        r, g, b = ((packed >> 16) & 255) / 255.0, ((packed >> 8) & 255) / 255.0, (packed & 255) / 255.0
        return np.array([r, g, b], dtype=np.float32)

    @staticmethod
    def _points_to_float_array(points_gen, ordered_names) -> np.ndarray:
        raw = np.array(list(points_gen))
        if raw.size == 0: return np.zeros((0, len(ordered_names)), dtype=np.float32)
        if raw.dtype.names is not None:
            cols = [raw[name].astype(np.float32) for name in ordered_names]
            return np.stack(cols, axis=1)
        arr = np.asarray(raw, dtype=np.float32)
        return arr.reshape(-1, len(ordered_names)) if arr.ndim == 1 else arr

    def _adapt_point_cloud_shape(self, pc_array: np.ndarray) -> np.ndarray:
        cur_feat_dim = pc_array.shape[1]
        if cur_feat_dim > self.pc_feature_dim:
            pc_array = pc_array[:, :self.pc_feature_dim]
        elif cur_feat_dim < self.pc_feature_dim:
            pad = np.zeros((pc_array.shape[0], self.pc_feature_dim - cur_feat_dim), dtype=np.float32)
            pc_array = np.concatenate([pc_array, pad], axis=1)

        cur_points = pc_array.shape[0]
        if cur_points > self.pc_num_points:
            idx = np.random.choice(cur_points, self.pc_num_points, replace=False)
            pc_array = pc_array[idx]
        elif cur_points < self.pc_num_points:
            pad = np.zeros((self.pc_num_points - cur_points, self.pc_feature_dim), dtype=np.float32)
            pc_array = np.concatenate([pc_array, pad], axis=0)
        return pc_array.astype(np.float32, copy=False)

    @staticmethod
    def _pose_to_array(pose: Pose) -> np.ndarray:
        p = pose.position
        q = pose.orientation
        # Keep quaternion order as xyzw to match training data and IK action format.
        return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)

    def synced_obs_cb(self, pose_msg: Pose, pc_msg: PointCloud2):
        if not self.collecting_obs:
            return

        state = self._pose_to_array(pose_msg)
        pc = self._pointcloud_to_array(pc_msg)
        if pc is None:
            return

        self.obs_state_window.append(state)
        self.obs_pc_window.append(pc)

        if len(self.obs_state_window) >= self.n_obs_steps:
            self.run_inference_from_window()

    def run_inference_from_window(self):
        obs_dict = {
            'agent_pos': torch.tensor(np.stack(self.obs_state_window), dtype=torch.float32).unsqueeze(0).to(self.device),
            'point_cloud': torch.tensor(np.stack(self.obs_pc_window), dtype=torch.float32).unsqueeze(0).to(self.device)
        }

        try:
            with torch.no_grad():
                pred = self.policy.predict_action(obs_dict)
            action_seq = pred['action'][0].detach().cpu().numpy()
            if action_seq.shape[0] < self.n_action_steps:
                self.get_logger().warn(
                    f"Predicted action steps {action_seq.shape[0]} < n_action_steps {self.n_action_steps}, skip."
                )
                return

            self.current_action_chunk = action_seq[:self.n_action_steps].copy()
            self.current_action_idx = 0
            self.collecting_obs = False
            self.infer_success_count += 1
            self.get_logger().info(
                f"Inference ready #{self.infer_success_count}: prepared {self.n_action_steps} actions."
            )
        except Exception as e:
            self.infer_fail_count += 1
            self.get_logger().error(f"Inference failed #{self.infer_fail_count}: {e}")

    def control_loop(self):
        if self.current_action_chunk is not None and self.current_action_idx < len(self.current_action_chunk):
            target_action = self.current_action_chunk[self.current_action_idx]
            self.current_action_idx += 1
            self.published_action_count += 1

            # 发布控制：前7维给IK位姿，第8维给夹爪
            self.publish_arm_control(target_action[:7])
            self.publish_gripper_control(target_action[7])

            now = time.time()
            if now - self.last_run_log_time > 2.0:
                self.get_logger().info(
                    f"Running: action {self.current_action_idx}/{self.n_action_steps}, "
                    f"total_published={self.published_action_count}, infer_fail={self.infer_fail_count}"
                )
                self.last_run_log_time = now

            if self.current_action_idx >= len(self.current_action_chunk):
                self.current_action_chunk = None
                self.current_action_idx = 0
                self.collecting_obs = True
                self.obs_state_window.clear()
                self.obs_pc_window.clear()
            return

        now = time.time()
        if self.collecting_obs and now - self.last_wait_log_time > 2.0:
            self.get_logger().info(
                f"Collecting synced obs: {len(self.obs_state_window)}/{self.n_obs_steps}"
            )
            self.last_wait_log_time = now

    def publish_arm_control(self, ik):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        # 注意：这里的 frame_id 使用了你要求的 "ee_link"，在实际机器人中，IK目标位姿通常是相对于基座坐标系的（例如 "base_link"），如有需要你可以自行调整
        msg.header.frame_id = "ee_link"

        msg.pose.position.x = float(ik[0])
        msg.pose.position.y = float(ik[1])
        msg.pose.position.z = float(ik[2])

        msg.pose.orientation.x = float(ik[3])  # qx
        msg.pose.orientation.y = float(ik[4])  # qy
        msg.pose.orientation.z = float(ik[5])  # qz
        msg.pose.orientation.w = float(ik[6])  # qw

        self.pub_arm_cmd.publish(msg)

    def publish_gripper_control(self, gripper_action):
        msg = HeaderFloat32()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "gripper_link"
        msg.data = float(gripper_action)
        self.pub_gripper_cmd.publish(msg)

    def destroy_node(self):
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    ckpt_path = Path("3D-Diffusion-Policy/data/outputs/simple_fruit_ik-real_simple_dp3-0309_seed0/checkpoints/latest.ckpt")
    cfg_path = ckpt_path.parent.parent / ".hydra" / "config.yaml"
    
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        return

    cfg = OmegaConf.load(cfg_path)
    workspace = TrainDP3Workspace(cfg, output_dir=str(ckpt_path.parent.parent))
    workspace.load_checkpoint(path=ckpt_path)

    policy = workspace.ema_model if cfg.training.use_ema and workspace.ema_model is not None else workspace.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device).eval()

    node = DP3InferenceNodeIK(cfg=cfg, policy=policy, device=device, n_obs_steps=int(cfg.n_obs_steps))

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
