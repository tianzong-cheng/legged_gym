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


device = "cuda"

path = "/home/rl/workspace/legged_gym/logs/wheel_legged_universal/Jun18_16-28-41_/model_80000.pt"
loaded_dict = torch.load(path)
# Filter out actor parameters
actor_state_dict = {
    k: v for k, v in loaded_dict["model_state_dict"].items() if k.startswith("actor.")
}

actor_critic = ActorCritic()
actor_critic.load_state_dict(actor_state_dict)

actor_critic.eval()
actor_critic.to(device)
policy = actor_critic.act_inference

# --------------------------------------------------

# Scaling factors
obs_scales = {
    "lin_vel": 2.0,
    "ang_vel": 0.25,
    "dof_pos": 1.0,
    "dof_vel": 0.05,
    "dof_acc": 0.0025,
    "height_measurements": 5.0,
    "torque": 0.05,
    "l": 5.0,
    "l_dot": 0.25,
}
commands_scale = torch.tensor(
    [
        obs_scales["lin_vel"],
        obs_scales["ang_vel"],
        obs_scales["height_measurements"],
    ],
    device=device,
    requires_grad=False,
)

base_ang_vel = (
    torch.tensor([0.5, 0.5, 0.5], device=device, requires_grad=False)
    * obs_scales["ang_vel"]
)
projected_gravity = torch.tensor([0, 0, 1], device=device, requires_grad=False)
theta_l = (
    torch.tensor([0, 0], device=device, requires_grad=False) * obs_scales["dof_pos"]
)
theta_l_dot = (
    torch.tensor([0, 0], device=device, requires_grad=False) * obs_scales["dof_vel"]
)
l = torch.tensor([0.3, 0.3], device=device, requires_grad=False) * obs_scales["l"]
l_dot = torch.tensor([0, 0], device=device, requires_grad=False) * obs_scales["l_dot"]
dof_pos = (
    torch.tensor([0.1, 0.2], device=device, requires_grad=False) * obs_scales["dof_pos"]
)
dof_vel = (
    torch.tensor([0.1, 0.2], device=device, requires_grad=False) * obs_scales["dof_vel"]
)
commands = (
    torch.tensor([0.5, 0.4, 0.3], device=device, requires_grad=False) * commands_scale
)
actions = torch.tensor([0, 0, 0, 0, 0, 0], device=device, requires_grad=False)

obs_buf = torch.zeros(27, device=device, requires_grad=False)
obs_buf = torch.cat(
    (
        base_ang_vel,
        projected_gravity,
        theta_l,
        theta_l_dot,
        l,
        l_dot,
        dof_pos,
        dof_vel,
        commands,
        actions,
    ),
    dim=-1,
)

print(obs_buf)

output = torch.zeros(6, device=device, requires_grad=False)
output = policy(obs_buf.detach())

print(output)
