import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        actor_hidden_dims = [128, 64, 32]
        activation = nn.ELU()
        mlp_input_dim_a = 27
        num_actions = 6

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1])
                )
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean


path = "/home/rl/workspace/legged_gym/logs/wheel_legged_universal/Jun18_16-28-41_/model_80000.pt"
loaded_dict = torch.load(path)
# Filter out actor parameters
actor_state_dict = {
    k: v for k, v in loaded_dict["model_state_dict"].items() if k.startswith("actor.")
}

actor_critic = ActorCritic()
actor_critic.load_state_dict(actor_state_dict)

actor_critic.eval()
# actor_critic.to(device)
policy = actor_critic.act_inference

# --------------------------------------------------

# Example float values
base_ang_vel = torch.tensor([0.5, 0.5, 0.5])
projected_gravity = torch.tensor([0, 0, 1])
theta_l = torch.tensor([0, 0])
theta_l_dot = torch.tensor([0, 0])
l = torch.tensor([0.3, 0.3])
l_dot = torch.tensor([0, 0])
dof_pos = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
dof_vel = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
commands = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1])
actions = torch.tensor([0, 0, 0, 0, 0, 0])

# Scaling factors (assuming these are also tensors)
obs_scales = {"ang_vel": 1.0, "dof_pos": 1.0, "dof_vel": 1.0, "l": 1.0, "l_dot": 1.0}
commands_scale = 1.0

# Convert float values to tensors and apply scaling

# Select specific columns from dof_pos and dof_vel
dof_pos_selected = dof_pos[[2, 5]] * obs_scales["dof_pos"]
dof_vel_selected = dof_vel[[2, 5]] * obs_scales["dof_vel"]
commands_scaled = commands[:3] * commands_scale

# Concatenate all tensors along the last dimension
obs_buf = torch.cat(
    (
        base_ang_vel,
        projected_gravity,
        theta_l,
        theta_l_dot,
        l,
        l_dot,
        dof_pos_selected,
        dof_vel_selected,
        commands_scaled,
        actions,
    ),
    dim=-1,
)

print(obs_buf)

actions = policy(obs_buf.detach())

print(actions)
