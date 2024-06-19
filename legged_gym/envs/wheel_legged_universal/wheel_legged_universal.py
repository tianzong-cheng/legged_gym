from legged_gym.envs.wheel_legged.wheel_legged import WheelLegged

import torch


class WheelLeggedUniversal(WheelLegged):
    def leg_forward_kinematics(self, theta_hip, theta_knee):
        wheel_x = -self.cfg.parameter.leg.l_thigh * torch.sin(
            self.pi - theta_hip
        ) + self.cfg.parameter.leg.l_shank * torch.cos(
            -theta_hip + theta_knee + self.pi / 2
        )
        wheel_y = -self.cfg.parameter.leg.l_thigh * torch.cos(
            theta_hip
        ) + self.cfg.parameter.leg.l_shank * torch.sin(
            -theta_hip + theta_knee + self.pi / 2
        )
        l = torch.sqrt(wheel_x**2 + wheel_y**2)
        theta_l = torch.arctan2(wheel_x, wheel_y)
        return l, theta_l

    def leg_observer(self):
        # Leg forward kinematics
        self.theta_hip = torch.cat(
            (-self.dof_pos[:, 0].unsqueeze(1), -self.dof_pos[:, 3].unsqueeze(1)), dim=1
        )
        self.theta_knee = torch.cat(
            (
                (self.dof_pos[:, 1] + 0.4396).unsqueeze(1),
                (self.dof_pos[:, 4] + 0.4396).unsqueeze(1),
            ),
            dim=1,
        )
        theta_hip_dot = torch.cat(
            (-self.dof_vel[:, 0].unsqueeze(1), -self.dof_vel[:, 3].unsqueeze(1)), dim=1
        )
        theta_knee_dot = torch.cat(
            (self.dof_vel[:, 1].unsqueeze(1), self.dof_vel[:, 4].unsqueeze(1)), dim=1
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
        heights = (
            self.root_states[:, 2].unsqueeze(1) * self.obs_scales.height_measurements
        )
        self.privileged_obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.obs_buf,
                self.last_actions,
                self.last_last_actions,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.dof_acc * self.obs_scales.dof_acc,
                heights,
                self.torques * self.obs_scales.torque,
                self.friction_coeffs.view(self.num_envs, 1),
                self.slip_left.unsqueeze(1),
                self.slip_right.unsqueeze(1),
            ),
            dim=-1,
        )
        # add noise if needed
        # Randomly generate noise between -noise_scale and noise_scale
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

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
        self.torque_wheel = torch.clip(self.torque_wheel, -5, 5)

        t_hip, t_knee = self.VMC(
            self.force_leg + self.cfg.control.f_feedforward, self.torque_leg
        )

        torques = torch.cat(
            (
                -t_hip[:, 0].unsqueeze(1),
                t_knee[:, 0].unsqueeze(1),
                self.torque_wheel[:, 0].unsqueeze(1),
                -t_hip[:, 1].unsqueeze(1),
                t_knee[:, 1].unsqueeze(1),
                self.torque_wheel[:, 1].unsqueeze(1),
            ),
            axis=1,
        )

        # TODO: Scale?
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def VMC(self, F, T):
        A = self.cfg.parameter.leg.l_thigh
        B = self.cfg.parameter.leg.l_shank

        theta_shank = -self.theta_hip + self.theta_knee + self.pi / 2

        j11 = -A * torch.cos(self.theta_hip) + B * torch.sin(theta_shank)
        j12 = -B * torch.sin(theta_shank)
        j21 = A * torch.sin(self.theta_hip) - B * torch.cos(theta_shank)
        j22 = B * torch.cos(theta_shank)

        t_hip = (j11 * torch.sin(self.theta_l) + j21 * torch.cos(self.theta_l)) * F + (
            j11 * torch.cos(self.theta_l) - j21 * torch.sin(self.theta_l)
        ) / self.l * T
        t_knee = (j12 * torch.sin(self.theta_l) + j22 * torch.cos(self.theta_l)) * F + (
            j12 * torch.cos(self.theta_l) - j22 * torch.sin(self.theta_l)
        ) / self.l * T

        return t_hip, t_knee
