import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.helpers import class_to_dict
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from .wheel_legged_config import WheelLeggedCfg


class WheelLegged(LeggedRobot):
    def __init__(
        self, cfg: WheelLeggedCfg, sim_params, physics_engine, sim_device, headless
    ):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2

    def leg_forward_kinematics(self, theta_hip, theta_knee):
        wheel_x = self.cfg.parameter.leg.l_thigh * torch.cos(
            theta_hip
        ) + self.cfg.parameter.leg.l_shank * torch.cos(theta_hip + theta_knee)
        wheel_y = self.cfg.parameter.leg.l_thigh * torch.sin(
            theta_hip
        ) + self.cfg.parameter.leg.l_shank * torch.sin(theta_hip + theta_knee)
        l = torch.sqrt(wheel_x**2 + wheel_y**2)
        theta_l = torch.arctan2(wheel_y, wheel_x) - self.pi / 2
        return l, theta_l

    def compute_observations(self):
        """
        3   base_ang_vel: Body IMU feedback
        3   projected_gravity: Body IMU feedback
        2   theta_l
        2   theta_l_dot
        2   l
        2   l_dot
        2   dof_pos (wheels)
        2   dof_vel (wheels)
        3   commands: Target vx, vy, wz
        6   actions: Motor target position
        Total: 27
        """
        # Leg forward kinematics
        self.theta_hip = torch.cat(
            (self.dof_pos[:, 0].unsqueeze(1), -self.dof_pos[:, 3].unsqueeze(1)), dim=1
        )
        self.theta_knee = torch.cat(
            (
                (self.dof_pos[:, 1] + self.pi / 2).unsqueeze(1),
                (-self.dof_pos[:, 4] + self.pi / 2).unsqueeze(1),
            ),
            dim=1,
        )
        theta_hip_dot = torch.cat(
            (self.dof_vel[:, 0].unsqueeze(1), -self.dof_vel[:, 3].unsqueeze(1)), dim=1
        )
        theta_knee_dot = torch.cat(
            (self.dof_vel[:, 1].unsqueeze(1), -self.dof_vel[:, 4].unsqueeze(1)), dim=1
        )

        self.l, self.theta_l = self.leg_forward_kinematics(
            self.theta_hip, self.theta_knee
        )

        # Predict l and theta_l to calculate l_dot and theta_l_dot
        dt = 0.001
        l_hat, theta_l_hat = self.leg_forward_kinematics(
            self.theta_hip + theta_hip_dot * dt, self.theta_knee + theta_knee_dot * dt
        )
        self.l_dot = (l_hat - self.l) / dt
        self.theta_l_dot = (theta_l_hat - self.theta_l) / dt

        # Load observation buffer
        self.obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.theta_l * self.obs_scales.dof_pos,
                self.theta_l_dot * self.obs_scales.dof_vel,
                self.l * self.obs_scales.l,
                self.l_dot * self.obs_scales.l_dot,
                self.dof_pos[:, [2, 5]] * self.obs_scales.dof_pos,
                self.dof_vel[:, [2, 5]] * self.obs_scales.dof_vel,
                self.commands[:, :3] * self.commands_scale,
                self.actions,
            ),
            dim=-1,
        )
        # add noise if needed
        # Randomly generate noise between -noise_scale and noise_scale
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (
            (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)
        # Calulate ang_vel_yaw from heading
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 1] = torch.clip(
                self.cfg.commands.kp_follow * wrap_to_pi(self.commands[:, 3] - heading),
                self.cfg.commands.ranges.ang_vel_yaw[0],
                self.cfg.commands.ranges.ang_vel_yaw[1],
            )

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (
            self.common_step_counter % self.cfg.domain_rand.push_interval == 0
        ):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["ang_vel_yaw"][0],
            self.command_ranges["ang_vel_yaw"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(
            self.command_ranges["height"][0],
            self.command_ranges["height"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

    def _compute_torques(self, actions):
        """
        actions[:, 0]   theta_l reference
        actions[:, 1]   l reference
        actions[:, 2]   theta_w_dot reference
        VMC: [t_v, F_v] -> [t_hip, t_knee]
        """
        theta_l_ref = (
            torch.cat(
                (
                    (actions[:, 0]).unsqueeze(1),
                    (actions[:, 3]).unsqueeze(1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_theta_l
        )
        l_ref = (
            torch.cat(
                (
                    (actions[:, 1]).unsqueeze(1),
                    (actions[:, 4]).unsqueeze(1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_l
        ) + self.cfg.control.l_offset
        wheel_speed_ref = (
            torch.cat(
                (
                    (actions[:, 2]).unsqueeze(1),
                    (actions[:, 5]).unsqueeze(1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_vel
        )

        self.torque_leg = (
            self.cfg.control.kp_theta_l * (theta_l_ref - self.theta_l)
            - self.cfg.control.kd_theta_l * self.theta_l_dot
        )
        self.force_leg = (
            self.cfg.control.kp_l * (l_ref - self.l)
            - self.cfg.control.kd_l * self.l_dot
        )
        self.torque_wheel = self.d_gains[[2, 5]] * (
            wheel_speed_ref - self.dof_vel[:, [2, 5]]
        )
        t_hip, t_knee = self.VMC(
            self.force_leg + self.cfg.control.f_feedforward, self.torque_leg
        )

        torques = torch.cat(
            (
                t_hip[:, 0].unsqueeze(1),
                t_knee[:, 0].unsqueeze(1),
                self.torque_wheel[:, 0].unsqueeze(1),
                -t_hip[:, 1].unsqueeze(1),
                -t_knee[:, 1].unsqueeze(1),
                self.torque_wheel[:, 1].unsqueeze(1),
            ),
            axis=1,
        )

        # TODO: Scale?
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def VMC(self, F, T):
        theta_temp = self.theta_l + self.pi / 2
        j11 = self.cfg.parameter.leg.l_thigh * torch.sin(
            theta_temp - self.theta_hip
        ) - self.cfg.parameter.leg.l_shank * torch.sin(
            self.theta_hip + self.theta_knee - theta_temp
        )

        j12 = self.cfg.parameter.leg.l_thigh * torch.cos(
            theta_temp - self.theta_hip
        ) - self.cfg.parameter.leg.l_shank * torch.cos(
            self.theta_hip + self.theta_knee - theta_temp
        )
        j12 = j12 / self.l

        j21 = -self.cfg.parameter.leg.l_shank * torch.sin(
            self.theta_hip + self.theta_knee - theta_temp
        )

        j22 = -self.cfg.parameter.leg.l_shank * torch.cos(
            self.theta_hip + self.theta_knee - theta_temp
        )
        j22 = j22 / self.l

        t_hip = j11 * F - j12 * T
        t_knee = j21 * F - j22 * T

        return t_hip, t_knee

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(
            self.cfg.domain_rand.push_interval_s / self.dt
        )

    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.torques = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.d_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
            requires_grad=False,
        )  # TODO change this
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        self.l = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.l_dot = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.theta_l = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.theta_l_dot = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.theta_hip = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.theta_knee = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(
                        f"PD gain of joint {name} were not defined, setting them to zero"
                    )
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
