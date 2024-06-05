from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class WheelLeggedCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_observations = 27
        num_privileged_obs = None
        num_actions = 6  # 4 jonit motors + 2 wheel motors

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"

    class commands:
        curriculum = False
        max_curriculum = 1.0
        num_commands = 4  # lin_vel_x, ang_vel_yaw, height, heading
        resampling_time = 4.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            height = [0.15, 0.38]  # min max [m]
            heading = [-3.14, 3.14]

        kp_follow = 1.5

    class init_state:
        pos = [0.0, 0.0, 0.25]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "lf0_Joint": 0.5,
            "lf1_Joint": 0.35,
            "l_wheel_Joint": 0.0,
            "rf0_Joint": -0.5,
            "rf1_Joint": -0.35,
            "r_wheel_Joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # Reinforcement Learning is not good at high frequency control
        # So use position control instead of torque control
        control_type = "P"
        stiffness = {"joint": 0.0, "wheel": 0.0}
        damping = {"joint": 0.0, "wheel": 0.5}
        action_scale = 0.5  # Why do we need action scales?
        decimation = 4

        kp_theta_l = 50.0  # [N*m/rad]
        kd_theta_l = 3.0  # [N*m*s/rad]
        kp_l = 900.0  # [N/m]
        kd_l = 20.0  # [N*s/m]

        action_scale_theta_l = 0.5
        action_scale_l = 0.1
        action_scale_vel = 10.0

        l_offset = 0.175
        f_feedforward = 0.0

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/wheel_legged/urdf/wl.urdf"
        name = "WheelLegged"  # actor name
        foot_name = "None"  # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = ["lf", "rf", "base"]
        terminate_after_contacts_on = ["base"]
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = (
            False  # Some .obj meshes must be flipped from y-up to z-up
        )

        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            l = 1.0
            l_dot = 1.0

        clip_observations = 100.0
        clip_actions = 100.0

    class parameter:
        class leg:
            l_thigh = 0.15
            l_shank = 0.25


class WheelLeggedCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [256, 128, 64]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24  # per iteration
        max_iterations = 1500  # number of policy updates

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = "test"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
