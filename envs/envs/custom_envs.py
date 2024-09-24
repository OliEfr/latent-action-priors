import numpy as np
import os
from pathlib import Path
import pathlib
from gymnasium.spaces import Box
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco import (
    humanoid_v4,
    ant_v4,
    half_cheetah_v4,
)


def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class HumanoidCustom_v4(humanoid_v4.HumanoidEnv):
    def __init__(self, latent_action_space_dim=False, **kwargs):
        self.latent_action_space_dim = latent_action_space_dim
        humanoid_v4.HumanoidEnv.__init__(self, **kwargs)

    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        observation = self._get_obs()
        reward = rewards - ctrl_cost
        terminated = self.terminated
        info = {
            "task_reward": reward,
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info
    
    def do_simulation(self, ctrl, n_frames):
        """
        Step the simulation n number of frames and applying a control action.
        """
        self._step_mujoco_simulation(ctrl, n_frames)

    def _set_action_space(self):
        if self.latent_action_space_dim:
            bounds = np.full((self.latent_action_space_dim, 2), [-1.0, 1.0]).astype(
                np.float32
            )
        else:
            bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = Box(low=low, high=high, dtype=np.float32)
        return self.action_space


class HumanoidHasExpertDataCustom_v4(humanoid_v4.HumanoidEnv):

    def __init__(
        self,
        latent_action_space_dim=False,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self.latent_action_space_dim = latent_action_space_dim

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        obs_space = 376 + 1  # 1 for phase
        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(obs_space,), dtype=np.float64
            )
        else:
            raise ValueError(
                "exclude_current_positions_from_observation should be True"
            )
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(378,), dtype=np.float64
            )

        MujocoEnv.__init__(
            self,
            "humanoid.xml",
            5,
            observation_space=observation_space,
            default_camera_config={
                "trackbodyid": 1,
                "distance": 4.0,
                "lookat": np.array((0.0, 0.0, 2.0)),
                "elevation": -20.0,
            },
            **kwargs,
        )

        expert_data = np.load(
            "expert_demonstrations/Humanoid/Humanoid.npy",
            allow_pickle=True,
        ).item()

        self.expert_obs_cycle = expert_data["recorded_obs_cycle"].copy()

        self.step_ = 0
        self.phase = 0


    def imitating_joint_pos_reward(self):
        joint_pos = self.data.qpos[2:].copy()
        joint_pos_ref = self.expert_obs_cycle[self.step_ % len(self.expert_obs_cycle)][
            :22
        ]
        return np.exp(-np.linalg.norm(joint_pos_ref - joint_pos))

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        com_inertia = self.data.cinert.flat.copy()
        com_velocity = self.data.cvel.flat.copy()

        actuator_forces = self.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.data.cfrc_ext.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
                np.array([self.phase]),
            )
        )

    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        self.step_ += 1
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        task_reward_normalization = 25
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        ctrl_cost = self.control_cost(action)
        task_reward = forward_reward + healthy_reward - ctrl_cost
        task_reward_normalized = task_reward / task_reward_normalization

        imitating_joint_pos_reward = self.imitating_joint_pos_reward()
        imitation_reward = imitating_joint_pos_reward

        reward = 0.67 * task_reward_normalized + 0.33 * imitation_reward

        observation = self._get_obs()
        terminated = self.terminated
        info = {
            "style_reward": imitation_reward,
            "joint_pos_reward": imitating_joint_pos_reward,
            "task_reward": task_reward,
            "task_reward_normalized": task_reward_normalized,
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info
    
    def do_simulation(self, ctrl, n_frames):
        """
        Step the simulation n number of frames and applying a control action.
        """
        self._step_mujoco_simulation(ctrl, n_frames)

    def reset_model(self):
        self.step_ = 0
        self.phase = 0

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def _set_action_space(self):
        if self.latent_action_space_dim:
            bounds = np.full((self.latent_action_space_dim, 2), [-1.0, 1.0]).astype(
                np.float32
            )
        else:
            bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = Box(low=low, high=high, dtype=np.float32)
        return self.action_space


class AntHasExpertDataCustom_v4(ant_v4.AntEnv):
    def __init__(self, latent_action_space_dim=False, **kwargs):

        self.latent_action_space_dim = latent_action_space_dim

        project_dir = Path(__file__).parent.parent.parent
        self.xml_file = str(
            project_dir
            / "envs/envs/assets/ant.xml"
        )

        self._ctrl_cost_weight = 0.5
        self._use_contact_forces = False
        self._contact_cost_weight = 5e-4
        self._healthy_reward = 1.0
        self._terminate_when_unhealthy = True
        self._healthy_z_range = (0.2, 1.0)
        self._contact_force_range = (-1.0, 1.0)
        self._reset_noise_scale = 0.1
        self._exclude_current_positions_from_observation = True

        utils.EzPickle.__init__(
            self,
            self.xml_file,
            self._ctrl_cost_weight,
            self._use_contact_forces,
            self._contact_cost_weight,
            self._healthy_reward,
            self._terminate_when_unhealthy,
            self._healthy_z_range,
            self._contact_force_range,
            self._reset_noise_scale,
            self._exclude_current_positions_from_observation,
        )

        obs_shape = 27 + 1  # 1 for phase
        if not self._exclude_current_positions_from_observation:
            raise ValueError(
                "exclude_current_positions_from_observation should be True"
            )
            obs_shape += 2
        if self._use_contact_forces:
            raise ValueError("use_contact_forces should be False")
            obs_shape += 84

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            self.xml_file,
            5,
            observation_space=observation_space,
            default_camera_config={
                "distance": 4.0,
            },
            **kwargs,
        )

        expert_data = np.load(
            "expert_demonstrations/Ant-v4.npy",
            allow_pickle=True,
        ).item()

        self.expert_obs_cycle = expert_data["recorded_obs_single_cycle"].copy()

        self.step_ = 0
        self.phase = 0


    def imitating_joint_pos_reward(self):
        joint_pos = self.data.qpos[2:].copy()
        joint_pos_ref = self.expert_obs_cycle[self.step_ % len(self.expert_obs_cycle)][
            :13
        ]
        return np.exp(-np.linalg.norm(joint_pos_ref - joint_pos))

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        self.step_ += 1
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        task_reward_normalization = 2
        forward_reward = x_velocity
        healthy_reward = self.healthy_reward
        costs = ctrl_cost = self.control_cost(action)
        task_reward = forward_reward + healthy_reward - costs
        task_reward_normalized = task_reward / task_reward_normalization

        imitating_joint_pos_reward = self.imitating_joint_pos_reward()
        imitation_reward = imitating_joint_pos_reward

        rewards = 0.67 * task_reward_normalized + 0.33 * imitation_reward

        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "style_reward": imitation_reward,
            "joint_pos_reward": imitating_joint_pos_reward,
            "task_reward": task_reward,
            "task_reward_normalized": task_reward_normalized,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        if self.render_mode == "human":
            self.render()
        return observation, rewards, terminated, False, info

    def do_simulation(self, ctrl, n_frames):
        """
        Step the simulation n number of frames and applying a control action.
        """
        self._step_mujoco_simulation(ctrl, n_frames)
    
    def _set_action_space(self):
        if self.latent_action_space_dim:
            bounds = np.full((self.latent_action_space_dim, 2), [-1.0, 1.0]).astype(
                np.float32
            )
        else:
            bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def reset_model(self):
        super().reset_model()

        self.step_ = 0
        self.phase = 0

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self._use_contact_forces:
            raise ValueError("use_contact_forces should be False")
            contact_force = self.contact_forces.flat.copy()
            return np.concatenate((position, velocity, contact_force))
        else:
            return np.concatenate((position, velocity, np.array([self.phase])))


class AntCustom_v4(ant_v4.AntEnv):
    def __init__(self, latent_action_space_dim=False, **kwargs):
        self.latent_action_space_dim = latent_action_space_dim
        ant_v4.AntEnv.__init__(self, **kwargs)

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        costs = ctrl_cost = self.control_cost(action)

        terminated = self.terminated
        observation = self._get_obs()

        if self._use_contact_forces:
            raise ValueError("use_contact_forces should be False")
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        reward = rewards - costs

        info = {
            "task_reward": reward,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info
    
    def do_simulation(self, ctrl, n_frames):
        """
        Step the simulation n number of frames and applying a control action.
        """
        self._step_mujoco_simulation(ctrl, n_frames)

    def _set_action_space(self):
        if self.latent_action_space_dim:
            bounds = np.full((self.latent_action_space_dim, 2), [-1.0, 1.0]).astype(
                np.float32
            )
        else:
            bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = Box(low=low, high=high, dtype=np.float32)
        return self.action_space


class HalfCheetahHasExpertDataCustom_v4(half_cheetah_v4.HalfCheetahEnv):
    def __init__(
        self,
        latent_action_space_dim=False,
        xml_file="envs/envs/assets/hc.xml",
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        self.latent_action_space_dim = latent_action_space_dim

        proj_root = pathlib.Path(__file__).parent.parent.parent
        xml_file = os.path.join(proj_root, xml_file)

        utils.EzPickle.__init__(
            self,
            xml_file,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        obs_shape = 17 + 1  # 1 for phase
        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
            )
        else:
            raise ValueError(
                "exclude_current_positions_from_observation should be True"
            )
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64
            )

        MujocoEnv.__init__(
            self,
            xml_file,
            5,
            observation_space=observation_space,
            default_camera_config={
                "distance": 4.0,
            },
            **kwargs,
        )

        expert_data = np.load(
            "expert_demonstrations/HalfCheetah-v4.npy",
            allow_pickle=True,
        ).item()

        self.expert_obs_cycle = expert_data["recorded_obs_single_cycle"].copy()

        self.step_ = 0
        self.phase = 0

    def imitating_joint_pos_reward(self):
        joint_pos = self.data.qpos[1:].copy()
        joint_pos_ref = self.expert_obs_cycle[self.step_ % len(self.expert_obs_cycle)][
            :8
        ]
        return np.exp(-np.linalg.norm(joint_pos_ref - joint_pos))

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        self.step_ += 1
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        task_reward_normalization = 2
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self.control_cost(action)
        task_reward = forward_reward - ctrl_cost
        task_reward_normalized = task_reward / task_reward_normalization

        imitating_joint_pos_reward = self.imitating_joint_pos_reward()
        imitation_reward = imitating_joint_pos_reward

        reward = 0.67 * task_reward_normalized + 0.33 * imitation_reward

        observation = self._get_obs()
        terminated = False
        info = {
            "style_reward": imitation_reward,
            "joint_pos_reward": imitating_joint_pos_reward,
            "task_reward": task_reward,
            "task_reward_normalized": task_reward_normalized,
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info
    
    def do_simulation(self, ctrl, n_frames):
        """
        Step the simulation n number of frames and applying a control action.
        """
        self._step_mujoco_simulation(ctrl, n_frames)

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate(
            (position, velocity, np.array([self.phase]))
        ).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        self.step_ = 0
        self.phase = 0

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def _set_action_space(self):
        if self.latent_action_space_dim:
            bounds = np.full((self.latent_action_space_dim, 2), [-1.0, 1.0]).astype(
                np.float32
            )
        else:
            bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = Box(low=low, high=high, dtype=np.float32)
        return self.action_space


class HalfCheetahCustom_v4(half_cheetah_v4.HalfCheetahEnv):
    def __init__(self, latent_action_space_dim=False, **kwargs):
        self.latent_action_space_dim = latent_action_space_dim
        half_cheetah_v4.HalfCheetahEnv.__init__(self, **kwargs)

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        terminated = False
        info = {
            "task_reward": reward,
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info
    
    def do_simulation(self, ctrl, n_frames):
        """
        Step the simulation n number of frames and applying a control action.
        """
        self._step_mujoco_simulation(ctrl, n_frames)

    def _set_action_space(self):
        if self.latent_action_space_dim:
            bounds = np.full((self.latent_action_space_dim, 2), [-1.0, 1.0]).astype(
                np.float32
            )
        else:
            bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = Box(low=low, high=high, dtype=np.float32)
        return self.action_space
