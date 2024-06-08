from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class WheelLeggedCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_observations = 27
        num_privileged_obs = 68
        num_actions = 6  # 4 jonit motors + 2 wheel motors

    class terrain:
        mesh_type = "trimesh"  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        # rough terrain only:
        measure_heights = True
        measured_points_x = [
            -0.8,
            -0.7,
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
        ]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 3  # starting curriculum state
        terrain_length = 10.0
        terrain_width = 10.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.3, 0.4, 0.0, 0.3, 0.0]
        # trimesh only:
        slope_treshold = (
            0.75  # slopes above this threshold will be corrected to vertical surfaces
        )

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 3.0
        num_commands = 4  # lin_vel_x, ang_vel_yaw, height, heading
        resampling_time = 5.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-6.28, 6.28]  # min max [rad/s]
            height = [0.1, 0.25]  # min max [m]
            heading = [-3.14, 3.14]

        kp_follow = 4

    class init_state(LeggedRobotCfg.init_state):
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
        decimation = 2

        kp_theta_l = 50.0  # [N*m/rad]
        kd_theta_l = 3.0  # [N*m*s/rad]
        kp_l = 900.0  # [N/m]
        kd_l = 20.0  # [N*s/m]

        action_scale_theta_l = 0.5
        action_scale_l = 0.1
        action_scale_vel = 10.0

        l_offset = 0.175
        f_feedforward = 40.0

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/wheel_legged/urdf/wl.urdf"
        name = "WheelLegged"  # actor name
        foot_name = "None"  # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = ["lf", "rf", "base"]
        terminate_after_contacts_on = ["base"]
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
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

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.5]
        randomize_base_mass = True
        added_mass_range = [-3.0, 3.0]
        push_robots = True
        push_interval_s = 7
        max_push_vel_xy = 2

    class rewards(LeggedRobotCfg.rewards):
        class scales:
            tracking_lin_vel = 1.0
            tracking_lin_vel_enhance = 1.0
            tracking_ang_vel = 1.0

            base_height = 1.0
            nominal_state = -0.1
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = 0.0
            orientation_exponential = 1.0

            dof_vel = -5e-5
            dof_acc = -2.5e-7
            joint_torques = -0.0001
            wheel_torques = -0.0001
            action_rate = -0.01
            action_smooth = -0.01

            collision = -1.0
            dof_pos_limits = -1.0
            theta_limit = -1.0

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = (
            0.97  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.18
        max_contact_force = 100.0  # forces above this value are penalized

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            height_measurements = 5.0
            torque = 0.05
            l = 5.0
            l_dot = 0.25

        clip_observations = 100.0
        clip_actions = 100.0

    class parameter:
        class leg:
            l_thigh = 0.15
            l_shank = 0.25


class WheelLeggedCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.5
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [256, 128, 64]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
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
        desired_kl = 0.005
        max_grad_norm = 1.0

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 48  # per iteration
        max_iterations = 2000  # number of policy updates

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = "wheel_legged"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
