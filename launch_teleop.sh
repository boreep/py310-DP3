#!/bin/bash

# 1. 激活 Conda 环境 (确保路径对)
source /home/hxd/miniconda3/bin/activate XRobot_py_env

# 2. Source ROS2 环境
source /opt/ros/humble/setup.bash

# 3. Source 你的工作空间
source /home/hxd/rm140_ws/install/setup.bash

# 4. 【核心黑魔法】设置混合库路径
# 解释：优先使用 Conda 的 lib (解决 OpenSSL)，其次使用 ROS/System 的 lib (解决 rclpy)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 5. 打印一下确认环境（调试用，稳定后可注释）
echo "Using Python: $(which python)"
echo "LD_LIBRARY_PATH head: $(echo $LD_LIBRARY_PATH | cut -d: -f1)"

# 6. 运行你的 Python 脚本，并传递所有参数 ($@)
python /home/hxd/code/My_Xrobot/scripts/teleop_robot.py "$@"