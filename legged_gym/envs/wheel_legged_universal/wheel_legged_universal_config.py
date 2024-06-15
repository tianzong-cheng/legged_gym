from legged_gym.envs.wheel_legged.wheel_legged_config import (
    WheelLeggedCfg,
    WheelLeggedCfgPPO,
)


class WheelLeggedUniversalCfg(WheelLeggedCfg):
    class commands(WheelLeggedCfg.commands):
        curriculum = True
        max_curriculum = 3.0
        num_commands = 4  # lin_vel_x, ang_vel_yaw, height, heading
        resampling_time = 5.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges(WheelLeggedCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-6.28, 6.28]  # min max [rad/s]
            height = [0.15, 0.3]  # min max [m]
            heading = [-3.14, 3.14]

        kp_follow = 4

    class init_state(WheelLeggedCfg.init_state):
        default_joint_angles = {  # target angles when action = 0.0
            "left_hip": -1.70,
            "left_knee": 0.28,
            "left_wheel_axis": 0.0,
            "right_hip": -1.70,
            "right_knee": 0.28,
            "right_wheel_axis": 0.0,
        }

    class control(WheelLeggedCfg.control):
        # Reinforcement Learning is not good at high frequency control
        # So use position control instead of torque control
        control_type = "P"
        stiffness = {"joint": 0.0, "wheel": 0.0}
        damping = {"joint": 0.0, "wheel": 0.5}
        action_scale = 0.5  # Why do we need action scales?
        decimation = 2

        kp_theta_l = 50.0  # [N*m/rad]
        kd_theta_l = 3.0  # [N*m*s/rad]
        kp_l = 900.0  # [N/m]
        kd_l = 20.0  # [N*s/m]

        action_scale_theta_l = 0.5
        action_scale_l = 0.1
        action_scale_vel = 10.0

        l_offset = 0.2
        f_feedforward = 100.0

    class asset(WheelLeggedCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/wheel_legged_universal/urdf/universal.urdf"
        penalize_contacts_on = ["shank", "thigh", "base"]
        terminate_after_contacts_on = ["base"]

    class rewards(WheelLeggedCfg.rewards):
        class scales:
            tracking_lin_vel = 1.0
            tracking_lin_vel_enhance = 1.0
            tracking_ang_vel = 1.0

            base_height = 1.0
            nominal_state = -1
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = 0.0
            orientation_exponential = 1.0

            dof_vel = -5e-5
            dof_acc = -8e-7
            joint_torques = -0.0001
            wheel_torques = -0.0001
            action_rate = -0.01
            action_smooth = -0.01

            collision = -1.0
            dof_pos_limits = -1.0
            theta_limit = -1.0

    class parameter:
        class leg:
            l_thigh = 0.25
            l_shank = 0.3


class WheelLeggedUniversalCfgPPO(WheelLeggedCfgPPO):
    class runner(WheelLeggedCfgPPO.runner):
        experiment_name = "wheel_legged_universal"
