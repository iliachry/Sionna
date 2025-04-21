import torch
import torch.nn as nn
import snntorch as snn
import snntorch.functional as SF
import snntorch.spikeplot as splt
import snntorch.surrogate as surrogate

from stable_baselines3.common.policies import ActorCriticPolicy

from typing import Callable, Tuple

from gymnasium import spaces


class CustomExtractorSNN(nn.Module):
    def __init__(self, feature_dim, action_size, last_layer_dim_vf=64, hidden_dim=64, timesteps=10):
        super().__init__()
        self.timesteps = timesteps

        self.latent_dim_pi = action_size
        self.latent_dim_vf = last_layer_dim_vf

        # Define spike neuron layer
        self.spike_grad = surrogate.fast_sigmoid(slope=25)

        # Policy SNN

        beta_in = torch.rand(hidden_dim)
        thr_in = torch.rand(hidden_dim)

        self.fc1_pi = nn.Linear(feature_dim, hidden_dim)
        self.lif1_pi = snn.Leaky(beta=beta_in, threshold=thr_in, learn_beta=True, spike_grad=self.spike_grad)

        self.fc2_pi = nn.Linear(hidden_dim, hidden_dim)
        self.lif2_pi = snn.Leaky(beta=beta_in, threshold=thr_in, learn_beta=True, spike_grad=self.spike_grad)

        beta_out = torch.rand(1)

        self.fc_out_pi = nn.Linear(hidden_dim, action_size)
        self.lif_out_pi = snn.Leaky(beta=beta_out, threshold=1.0, spike_grad=self.spike_grad, learn_beta=True, reset_mechanism="none")


        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, last_layer_dim_vf),
            nn.ReLU()
        )

    def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        mem1 = self.lif1_pi.init_leaky()
        mem2 = self.lif2_pi.init_leaky()
        mem3 = self.lif_out_pi.init_leaky()

        for _ in range(self.timesteps):
            cur = self.fc1_pi(x)
            spk1, mem1 = self.lif1_pi(cur, mem1)
            cur = self.fc2_pi(spk1)
            spk2, mem2 = self.lif2_pi(cur, mem1)
            cur = self.fc_out_pi(spk2)
            spk3, mem3 = self.lif_out_pi(cur, mem2)
        scaled_mem = torch.tanh(mem3) * torch.pi
        return scaled_mem

    def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_net(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(x), self.forward_critic(x)
    


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        kwargs["ortho_init"] = False
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomExtractorSNN(self.features_dim, self.action_space.shape[0])

    def _build(self, lr_schedule):

        self._build_mlp_extractor()

        self.action_net = nn.Identity()
        self.log_std = nn.Parameter(torch.zeros(self.action_space.shape[0]), requires_grad=True)
        
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)