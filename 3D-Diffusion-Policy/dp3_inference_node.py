import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import time
from pathlib import Path
from collections import deque
import numpy as np

import torch
from omegaconf import OmegaConf

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from message_filters import Subscriber, ApproximateTimeSynchronizer

# 导入你指定的自定义接口
from rm_ros_interfaces.msg import Jointpos
from my_interfaces.msg import HeaderFloat32

# 导入 DP3 的环境
from train import TrainDP3Workspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

class DP3InferenceNode(Node):
    def __init__(self, cfg, policy, device, n_obs_steps):
        super().__init__('dp3_inference_node')
        self.cfg = cfg
        self.policy = policy
        self.device = device
        self.n_obs_steps = n_obs_steps
        
        # --- 参数配置 ---
        self.control_rate = float(OmegaConf.select(cfg, "inference.control_rate", default=20.0))
        self.use_pc_color = bool(OmegaConf.select(cfg, "policy.use_pc_color", default=False))
        self.n_action_steps = int(OmegaConf.select(cfg, "n_action_steps", default=1))
        # exec_action_steps 决定了我们一次性执行多少步动作后再重新推理 (等同于参考代码的 action_horizon)
        self.exec_action_steps = int(OmegaConf.select(cfg, "inference.exec_action_steps", default=4))
        self.exec_action_steps = max(1, min(self.exec_action_steps, self.n_action_steps))
        
        action_shape = OmegaConf.select(cfg, "shape_meta.action.shape", default=[7])
        self.action_dim = int(action_shape[0]) if len(action_shape) > 0 else 7
        pc_shape = OmegaConf.select(cfg, "shape_meta.obs.point_cloud.shape", default=[1024, 3])
        self.pc_num_points = int(pc_shape[0]) if len(pc_shape) > 0 else 1024
        self.pc_feature_dim = int(pc_shape[1]) if len(pc_shape) > 1 else (6 if self.use_pc_color else 3)
        self.expected_joint_names = OmegaConf.select(cfg, "inference.joint_names", default=None)
        self.arm_joint_limits = OmegaConf.select(cfg, "inference.arm_joint_limits", default=None)
        self.gripper_range = OmegaConf.select(cfg, "inference.gripper_range", default=[0.0, 1.0])

        # --- 核心状态变量 ---
        self.current_action_chunk = None
        self.current_action_idx = 0
        
        # 滑动窗口：满了之后新数据会自动挤掉老数据，不需要手动 clear()
        self.obs_state_window = deque(maxlen=self.n_obs_steps)
        self.obs_pc_window = deque(maxlen=self.n_obs_steps)
        
        self.infer_success_count = 0
        self.infer_fail_count = 0
        self.published_action_count = 0

        # --- 使用 ROS2 近似时间同步订阅观测 ---
        self.sub_joint_state = Subscriber(self, JointState, 'right_arm/joint_states')
        self.sub_pointcloud = Subscriber(self, PointCloud2, 'camera/sampled_points')
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_joint_state, self.sub_pointcloud],
            queue_size=max(20, self.n_obs_steps * 2),
            slop=0.05
        )
        self.sync.registerCallback(self.synced_obs_cb)
        
        # --- 发布者 ---
        self.pub_arm_cmd = self.create_publisher(Jointpos, 'right_arm/rm_driver/movej_canfd_cmd', 10)
        self.pub_gripper_cmd = self.create_publisher(HeaderFloat32, 'right_arm/gripper_cmd', 10)
        
        # --- 定时器 ---
        self.timer = self.create_timer(1.0 / self.control_rate, self.control_loop)
        
        self.get_logger().info(
            f"DP3 Node Started. Control: {self.control_rate}Hz, "
            f"Actions per plan: {self.n_action_steps}, exec per replan: {self.exec_action_steps}."
        )

    def synced_obs_cb(self, joint_msg: JointState, pc_msg: PointCloud2):
        # 取消了所有的阻塞和收集状态判断。这里充当“后台边走边录”的角色。
        state = self._extract_state(joint_msg)
        if state is None:
            return
        pc = self._pointcloud_to_array(pc_msg)
        if pc is None:
            return

        # 永远把最新的数据塞进窗口
        self.obs_state_window.append(state)
        self.obs_pc_window.append(pc)

    def run_inference_from_window(self):
        # 直接使用当前窗口中最新的观测进行推理
        obs_dict = {
            'agent_pos': torch.tensor(np.stack(self.obs_state_window), dtype=torch.float32).unsqueeze(0).to(self.device),
            'point_cloud': torch.tensor(np.stack(self.obs_pc_window), dtype=torch.float32).unsqueeze(0).to(self.device)
        }

        try:
            with torch.no_grad():
                pred = self.policy.predict_action(obs_dict)
            action_seq = pred['action'][0].detach().cpu().numpy()
            
            if action_seq.shape[0] < self.n_action_steps:
                self.get_logger().warn("Predicted action steps too short, skip.")
                return False

            # 更新 action chunk 并重置索引
            self.current_action_chunk = action_seq[:self.exec_action_steps].copy() # 截取需要执行的步数
            self.current_action_idx = 0
            self.infer_success_count += 1
            return True
            
        except Exception as e:
            self.infer_fail_count += 1
            self.get_logger().error(f"Inference failed #{self.infer_fail_count}: {e}")
            return False

    def control_loop(self):
        # 1. 初始化检查：如果没有攒够所需的最少观测帧，跳过并等待
        if len(self.obs_state_window) < self.n_obs_steps:
            self.get_logger().info("Waiting for initial observations...", throttle_duration_sec=2.0)
            return

        # 2. 推理触发：如果没有动作，或者上一批动作已经执行完毕
        if self.current_action_chunk is None or self.current_action_idx >= self.exec_action_steps:
            success = self.run_inference_from_window()
            if not success:
                return  # 如果推理失败，这一帧不执行动作，等下个周期重试

        # 3. 动作下发：依次执行当前序列中的动作
        target_action = self.current_action_chunk[self.current_action_idx]
        self.current_action_idx += 1
        self.published_action_count += 1

        arm_action = target_action[:6]
        gripper_action = target_action[6]

        # 限位处理
        if self.arm_joint_limits is not None and len(self.arm_joint_limits) == 6:
            arm_action = np.clip(arm_action, 
                                 [lim[0] for lim in self.arm_joint_limits], 
                                 [lim[1] for lim in self.arm_joint_limits])
        if isinstance(self.gripper_range, (list, tuple)):
            gripper_action = float(np.clip(gripper_action, self.gripper_range[0], self.gripper_range[1]))

        # 发布控制指令
        self.publish_arm_control(arm_action)
        self.publish_gripper_control(gripper_action)

    # ---------------- 原始数据处理方法保持不变 ----------------
    def publish_arm_control(self, arm_action):
        msg = Jointpos()
        msg.joint = arm_action.tolist()
        msg.follow = False   
        msg.expand = 0.0
        self.pub_arm_cmd.publish(msg)

    def publish_gripper_control(self, gripper_action):
        msg = HeaderFloat32()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "gripper_link"
        msg.data = float(gripper_action)
        self.pub_gripper_cmd.publish(msg)

    def _extract_state(self, joint_msg: JointState) -> np.ndarray:
        # [代码省略，与你原先的逻辑完全一致]
        if self.expected_joint_names:
            name_to_idx = {name: i for i, name in enumerate(joint_msg.name)}
            try:
                idx = [name_to_idx[name] for name in self.expected_joint_names]
            except KeyError as e:
                return None
            vals = [joint_msg.position[i] for i in idx]
            return np.array(vals, dtype=np.float32)
        if len(joint_msg.position) < 6:
            return None
        return np.array(joint_msg.position[:6], dtype=np.float32)

    def _pointcloud_to_array(self, msg: PointCloud2) -> np.ndarray:
        # [代码省略，与你原先的逻辑完全一致]
        field_names = [f.name for f in msg.fields]
        points_gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        xyz_array = self._points_to_float_array(points_gen, ("x", "y", "z"))
        if len(xyz_array) == 0:
            return None
        pc_array = xyz_array
        return self._adapt_point_cloud_shape(pc_array)

    @staticmethod
    def _points_to_float_array(points_gen, ordered_names) -> np.ndarray:
        raw = np.array(list(points_gen))
        if raw.size == 0: return np.zeros((0, len(ordered_names)), dtype=np.float32)
        arr = np.asarray(raw, dtype=np.float32)
        return arr.reshape(-1, len(ordered_names)) if arr.ndim == 1 else arr

    def _adapt_point_cloud_shape(self, pc_array: np.ndarray) -> np.ndarray:
        # [代码省略，与你原先的逻辑完全一致]
        cur_points = pc_array.shape[0]
        if cur_points > self.pc_num_points:
            idx = np.random.choice(cur_points, self.pc_num_points, replace=False)
            pc_array = pc_array[idx]
        elif cur_points < self.pc_num_points:
            pad = np.zeros((self.pc_num_points - cur_points, self.pc_feature_dim), dtype=np.float32)
            pc_array = np.concatenate([pc_array, pad], axis=0)
        return pc_array.astype(np.float32, copy=False)


def main(args=None):
    rclpy.init(args=args)
    
    ckpt_path = Path("3D-Diffusion-Policy/data/outputs/real_simple_fruit-real_dp3-0309_seed0/checkpoints/epoch=0400-test_mean_score=-0.001.ckpt")
    cfg_path = ckpt_path.parent.parent / ".hydra" / "config.yaml"
    
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        return

    cfg = OmegaConf.load(cfg_path)
    workspace = TrainDP3Workspace(cfg, output_dir=str(ckpt_path.parent.parent))
    workspace.load_checkpoint(path=ckpt_path)

    policy = workspace.ema_model if cfg.training.use_ema and workspace.ema_model is not None else workspace.model
    device = torch.device("cuda")
    policy.to(device).eval()

    node = DP3InferenceNode(cfg=cfg, policy=policy, device=device, n_obs_steps=int(cfg.n_obs_steps))

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()