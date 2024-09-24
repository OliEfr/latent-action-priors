from collections import deque
import numpy as np
import torch
from gymnasium import ActionWrapper
import wandb


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ProjectActions(ActionWrapper):
    def __init__(
        self,
        env,
        action_space_shape,
        latent_action_space_dim,
        projector_path,
        projector_size,
        projector_type,
        full_actions_weight,
        action_log_freq=2048,  # standard n_steps for PPO in stable-baselines3
    ):
        super().__init__(env)

        assert (
            projector_size[1] == action_space_shape
        ), "Projector size [1] should match action space shape."

        self.action_space_shape = action_space_shape
        self.latent_action_space_dim = latent_action_space_dim
        self.projector_size = projector_size
        self.full_actions_weight = full_actions_weight

        self.step_ = 0

        # log actions
        # define our custom x axis metric
        # define which metrics will be plotted against it
        wandb.define_metric("actions/*", step_metric="action_log_step")
        self.action_log_freq = action_log_freq
        self.projected_actions_contribution_buffer = deque(maxlen=self.action_log_freq)
        self.projected_actions_contribution_weighted_buffer = deque(
            maxlen=self.action_log_freq
        )
        self.full_actions_contribution_buffer = deque(maxlen=self.action_log_freq)
        self.full_actions_contribution_weighted_buffer = deque(
            maxlen=self.action_log_freq
        )
        self.latent_actions_contribution_buffer = deque(maxlen=self.action_log_freq)

        if projector_type == "nonLin":
            scaling = 2  # always use scaling of 2
            latent_dim = projector_size[0]
            dim = action_space_shape
            self.projector = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, latent_dim * scaling),
                torch.nn.Tanh(),
                torch.nn.Linear(latent_dim * scaling, latent_dim * scaling),
                torch.nn.Tanh(),
                torch.nn.Linear(latent_dim * scaling, dim),
            ).double()
        else:
            raise ValueError("Unknown projector type.")
        self.projector.load_state_dict(torch.load(projector_path))
        print(
            "Loaded projector from ",
            projector_path,
        )

    def action(self, act):
        # act are clipped to [-1, 1]
        self.step_ += 1
        with torch.no_grad():
            # only projection
            if self.latent_action_space_dim == self.projector_size[0]:
                actions = (
                    self.projector(torch.tensor(act, dtype=torch.double))
                    .detach()
                    .numpy()
                )
            # projection and residual
            else:
                actions_projected = (
                    self.projector(
                        torch.tensor(act[: self.projector_size[0]], dtype=torch.double)
                    )
                    .detach()
                    .numpy()
                )  # first actions are latent actions
                actions_full = act[
                    self.projector_size[0] :
                ]  # last actions are full actions

                # logging
                self.projected_actions_contribution_buffer.append(
                    np.mean(np.abs(actions_projected))
                )
                self.projected_actions_contribution_weighted_buffer.append(
                    np.mean(np.abs(actions_projected))
                    * (1 - self.full_actions_weight)
                )
                self.full_actions_contribution_buffer.append(
                    np.mean(np.abs(actions_full))
                )
                self.full_actions_contribution_weighted_buffer.append(
                    np.mean(np.abs(actions_full)) * self.full_actions_weight
                )
                self.latent_actions_contribution_buffer.append(
                    np.mean(np.abs(act[: self.projector_size[0]]))
                )
                if self.step_ % self.action_log_freq == 0:
                    wandb.log(
                        {
                            "actions/projected_actions_contribution": np.mean(
                                self.projected_actions_contribution_buffer
                            ),
                            "actions/projected_actions_contribution_weighted": np.mean(
                                self.projected_actions_contribution_weighted_buffer
                            ),
                            "actions/full_actions_contribution": np.mean(
                                self.full_actions_contribution_buffer
                            ),
                            "actions/full_actions_contribution_weighted": np.mean(
                                self.full_actions_contribution_weighted_buffer
                            ),
                            "actions/latent_actions_contribution": np.mean(
                                self.latent_actions_contribution_buffer
                            ),
                            "action_log_step": self.step_,
                        },
                    )

            # naively add them together
            actions = (
                1 - self.full_actions_weight
            ) * actions_projected + self.full_actions_weight * actions_full
            return actions


class ProjectActions_multiUnitreeA1(ActionWrapper):
    def __init__(
        self,
        env,
        action_space_shape,
        latent_action_space_dim,
        projector_path,
        projector_size,
        projector_type,
        full_actions_weight,
        n_UnitreeA1,
    ):
        super().__init__(env)

        assert (
            action_space_shape % projector_size[1] == 0
        ), "Projector size [1] should match action space shape or a multiple of it."

        self.action_space_shape = action_space_shape
        self.latent_action_space_dim = latent_action_space_dim
        self.projector_size = projector_size
        self.full_actions_weight = full_actions_weight
        self.n_UnitreeA1 = n_UnitreeA1

        if projector_type == "nonLin":
            scaling = 2  # always use scaling of 2
            latent_dim = projector_size[0]
            dim = self.action_space_shape // self.n_UnitreeA1
            self.projector = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, latent_dim * scaling),
                torch.nn.Tanh(),
                torch.nn.Linear(latent_dim * scaling, latent_dim * scaling),
                torch.nn.Tanh(),
                torch.nn.Linear(latent_dim * scaling, dim),
            ).double()
        else:
            raise ValueError("Unknown projector type.")
        self.projector.load_state_dict(torch.load(projector_path))
        print(
            "Loaded projector from ",
            projector_path,
        )

    def action(self, act):
        # EFFICIENT IMPLEMENTATION (untested)
        # act = torch.tensor(act, dtype=torch.double).reshape(self.n_UnitreeA1, self.latent_action_space_dim // self.n_UnitreeA1)
        # with torch.no_grad():
        #     projected_actions = self.projector(act[:, :self.projector_size[0]])
        #     full_actions = (1 - self.projector_residual_weight) * projected_actions + self.projector_residual_weight * act[:, self.projector_size[0]:]
        #     return full_actions.flatten()

        # UNEFFICIENT IMPLEMENTATION (used for paper)
        with torch.no_grad():
            # only projection
            if self.latent_action_space_dim == self.projector_size[0]:
                raise ValueError("Require full action space residual.")
            # projection and residual
            else:
                singleUnitreeA1_latent_action_space_dim = (
                    self.latent_action_space_dim // self.n_UnitreeA1
                )
                singleUnitreeA1_action_space_dim = (
                    self.action_space_shape // self.n_UnitreeA1
                )
                actions_multiUnitreeA1 = np.zeros(self.action_space_shape)
                for i in range(self.n_UnitreeA1):
                    indx_offset = i * singleUnitreeA1_latent_action_space_dim
                    actions_projected = (
                        self.projector(
                            torch.tensor(
                                act[indx_offset : indx_offset + self.projector_size[0]],
                                dtype=torch.double,
                            )
                        )
                        .detach()
                        .numpy()
                    )
                    actions_full = act[
                        indx_offset
                        + self.projector_size[0] : indx_offset
                        + singleUnitreeA1_latent_action_space_dim
                    ]
                    actions_singleUnitreeA1 = (
                        (1 - self.full_actions_weight) * actions_projected
                        + self.full_actions_weight * actions_full
                    )
                    actions_multiUnitreeA1[
                        singleUnitreeA1_action_space_dim
                        * i : singleUnitreeA1_action_space_dim
                        * (i + 1)
                    ] = actions_singleUnitreeA1

            return actions_multiUnitreeA1
