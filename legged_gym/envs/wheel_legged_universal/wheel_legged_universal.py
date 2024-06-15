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
