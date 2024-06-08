# Notes on Reinforcement Learning Development

## Question Marks

### Q1

Why are the privileged observations needed?

```python
self.privileged_obs_buf = torch.cat(
    (
        self.base_lin_vel * self.obs_scales.lin_vel,
        self.obs_buf,
        self.last_actions[:, :, 0],
        self.last_actions[:, :, 1],
        self.dof_acc * self.obs_scales.dof_acc,
        heights,
        self.torques * self.obs_scales.torque,
        (self.base_mass - self.base_mass.mean()).view(selfnum_envs, 1),
        self.base_com,
        self.default_dof_pos - self.raw_default_dof_pos,
        self.friction_coef.view(self.num_envs, 1),
        self.restitution_coef.view(self.num_envs, 1),
    ),
    dim=-1,
)
```

### Q2

Why `10.0`?

```python
fail_buf = torch.any(
    torch.norm(
        self.contact_forces[:, self.termination_contact_indices, :], dim=-1
    )
    > 10.0,
    dim=1,
)
```

### Q3

```python
self.fail_buf *= fail_buf
self.fail_buf += fail_buf

self.fail_buf = fail_buf
```

## TODO

- [ ] `step()`
  - [ ] Action FIFO
  - [ ] Pushes robot in main loop
- [ ] `compute_dof_vel()`
  - [ ] Why?
- [ ] `post_physics_step()`
  - [ ] Refreshes rigid body states
    - [ ] Where are rigid body states used?
    - [ ] Bro, the tensor is not even acquired?
  - [ ] Uses differentiation
  - [ ] Calculates acceleration
- [ ] `leg_post_physics_step()`, `forward_kinematcis()`
  - [ ] **Calculates leg length and leg angle**
- [ ] `check_termination()`
  - [ ] z-axis gravity check
  - [ ] terrain boundary check
- [ ] `reset_idx()`
  - [ ] Resets additional variables added by user
  - [ ] What does `extras` do?
- [ ] `compute_reward()`
  - [ ] Reward clipping
- [ ] `_process_rigid_shape_props()`
  - [ ] Randomize restitution
- [ ] `_process_rigid_body_props()`
  - [ ] Randomize base center of mass
  - [ ] Randomize inertia
- [ ] `_post_physics_step_callback()`
  - [ ] `self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)`
  - [ ] `0.5` is changed to `1.5`, why?
- [ ] `_reset_dofs()`
  - [ ] Randomize default DOF position
- [ ] `_push_robots()`
  - [ ] Uses `self.gym.apply_rigid_body_force_tensors()`
  - [ ] Why?
- [ ] `_update_terrain_curriculum()`
  - [ ] How to set terrain level?
- Use position dot instead of velocity, why?
- [ ] `update_command_curriculum()`
  - [ ] Command curriculum
- [ ] `_get_noise_scale_vec()`
  - [ ] User added noises
- [ ] `_get_env_origins()`
- [ ] `pre_physics_step()`
  - [ ] Seems not used

# Notes 6/6

- `envs_steps_buf`
- `_push_robots()`
- `refresh_rigid_body_state_tensor()`, should be OK
- Still no curriculum
  - Command curriculum logic
- `fail_buf`

# Notes 6/8

- Terrain friction?
- URDF
- Fly
- Sudden loss in friction
