import numpy as np
from pathlib import Path

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"distance": 4.0, "elevation": -15.0}


class UnitreeA1Custom(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(
        self,
        latent_action_space_dim: bool = False,
        mode: str = "easy",
        target_speed_scale: float = 1.0,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            **kwargs,
        )

        self.latent_action_space_dim = latent_action_space_dim

        self.target_speed_scale = target_speed_scale

        self.mode = mode

        self.init_samples = np.load("./envs/envs/assets/assets/unitree_a1_traj_init_samples.npy")

        project_dir = Path(__file__).parent.parent.parent
        self.xml_path = str(
            project_dir
            / "envs/envs/assets/unitree_a1_torque.xml"
        )

        obs_shape = (
            18 + 16
        ) + 2  # 18 velocities, 16 positions (without x-y trunk position), 2 desired velocity
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )
       
        if mode == "easy":
            self.desired_velocity = np.array([0.23, 0.0]) * self.target_speed_scale
        elif mode == "hard":
            random_vector = np.random.rand(2)
            norm = np.linalg.norm(random_vector)
            self.desired_velocity = (random_vector / norm) * 0.23 * self.target_speed_scale
        else:
            raise ValueError("mode must be either 'easy' or 'hard'")


        MujocoEnv.__init__(
            self,
            self.xml_path,
            10,  # frame_skip
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            width=1920,
            height=1080,
            camera_name="track",
            **kwargs,
        )

    def step(self, action):

        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._has_fallen()

        info = {
            "task_reward": reward,
        }

        return obs, reward, terminated, False, info
    
    def do_simulation(self, ctrl, n_frames):
        """
        Step the simulation n number of frames and applying a control action.
        """
        self._step_mujoco_simulation(ctrl, n_frames)

    def _get_reward(self):

        # reward is mean difference of agent from target velocity
        dq_trunk_tx_id = self.model.joint(f"trunk_tx").id
        dq_trunk_ty_id = self.model.joint(f"trunk_ty").id

        curr_velocity_xy = np.array(
            [self.data.qvel[dq_trunk_tx_id], self.data.qvel[dq_trunk_ty_id]]
        )

        reward = np.exp(-5.0 * np.linalg.norm(curr_velocity_xy - self.desired_velocity))

        return reward

    def _has_fallen(self):

        # _has_fallen condition only depends on qpos
        qpos = self.data.qpos.flat.copy()

        q_trunk_tz_id = self.model.joint(f"trunk_tz").id
        q_trunk_list_id = self.model.joint(f"trunk_list").id
        q_trunk_tilt_id = self.model.joint(f"trunk_tilt").id

        trunk_height_condition = qpos[q_trunk_tz_id] < -0.24
        trunk_list_condition = (qpos[q_trunk_list_id] < -0.2793) or (
            qpos[q_trunk_list_id] > 0.2793
        )
        trunk_tilt_condition = (qpos[q_trunk_tilt_id] < -0.192) or (
            qpos[q_trunk_tilt_id] > 0.192
        )

        trunk_condition = (
            trunk_list_condition or trunk_tilt_condition or trunk_height_condition
        )

        return trunk_condition

    def _get_obs(self):

        position = self.data.qpos.flat.copy()[
            2:
        ]  # 2 avoids x-y position of trunk in the observation
        velocity = self.data.qvel.flat.copy()

        obs = np.concatenate((position, velocity, self.desired_velocity))

        return obs

    def reset_model(self):

        if self.mode == "hard":
            random_vector = np.random.rand(2)
            norm = np.linalg.norm(random_vector)
            self.desired_velocity = (random_vector / norm) * 0.23 * self.target_speed_scale

        rand_idx = np.random.choice(self.init_samples.shape[0])
        init_states = self.init_samples[rand_idx]
        init_qpos = init_states[:18]
        init_qvel = init_states[18:]

        self.set_state(init_qpos, init_qvel)

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


class UnitreeA1HasExpertDataCustom(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(
        self,
        latent_action_space_dim: bool = False,
        mode: str = "easy",
        target_speed_scale: float = 1.0,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            **kwargs,
        )

        self.latent_action_space_dim = latent_action_space_dim

        self.target_speed_scale = target_speed_scale

        self.mode = mode
        
        self.init_samples = np.load("./envs/envs/assets/assets/unitree_a1_traj_init_samples.npy")

        project_dir = Path(__file__).parent.parent.parent
        self.xml_path = str(
            project_dir
            / "envs/envs/assets/unitree_a1_torque.xml"
        )

        obs_shape = (
            18 + 16
        ) + 2  + 1 # 18 velocities, 16 positions (without x-y trunk position), 2 desired velocity; + 1 for phase
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        if mode == "easy":
            self.desired_velocity = np.array([0.23, 0.0]) * self.target_speed_scale
        elif mode == "hard":
            random_vector = np.random.rand(2)
            norm = np.linalg.norm(random_vector)
            self.desired_velocity = (random_vector / norm) * 0.23 * self.target_speed_scale
        else:
            raise ValueError("mode must be either 'easy' or 'hard'")


        MujocoEnv.__init__(
            self,
            self.xml_path,
            10,  # frame_skip
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            width=1920,
            height=1080,
            camera_name="track",
            **kwargs,
        )

        expert_data = np.load(
            "expert_demonstrations/Unitree_A1_H1/UnitreeA1.simple.perfect.npy",
            allow_pickle=True,
        ).item()
        self.expert_obs_cycle = expert_data["recorded_obs_cycle"].copy()

        self.step_ = 0
        self.phase = 0

    def imitating_joint_pos_reward(self):
        joint_pos = self.data.qpos[2:].copy()
        joint_pos_ref = self.expert_obs_cycle[self.step_ % len(self.expert_obs_cycle)][
            :16
        ]
        return np.exp(-np.linalg.norm(joint_pos_ref - joint_pos))

    def step(self, action):

        self.do_simulation(action, self.frame_skip)
        self.step_ += 1
        self.phase = self.step_ % len(self.expert_obs_cycle)

        obs = self._get_obs()
        
        # reward is mean difference of agent from target velocity
        dq_trunk_tx_id = self.model.joint(f"trunk_tx").id
        dq_trunk_ty_id = self.model.joint(f"trunk_ty").id

        curr_velocity_xy = np.array(
            [self.data.qvel[dq_trunk_tx_id], self.data.qvel[dq_trunk_ty_id]]
        )

        task_reward = np.exp(-5.0 * np.linalg.norm(curr_velocity_xy - self.desired_velocity))
        style_reward = self.imitating_joint_pos_reward()

        reward = 0.67 * task_reward + 0.33 * style_reward


        terminated = self._has_fallen()

        info = {
            "task_reward": task_reward,
            "style_reward": style_reward,
            "joint_pos_reward": style_reward,
        }

        return obs, reward, terminated, False, info
    
    def do_simulation(self, ctrl, n_frames):
        """
        Step the simulation n number of frames and applying a control action.
        """
        self._step_mujoco_simulation(ctrl, n_frames)

    def _has_fallen(self):

        # _has_fallen condition only depends on qpos
        qpos = self.data.qpos.flat.copy()

        q_trunk_tz_id = self.model.joint(f"trunk_tz").id
        q_trunk_list_id = self.model.joint(f"trunk_list").id
        q_trunk_tilt_id = self.model.joint(f"trunk_tilt").id

        trunk_height_condition = qpos[q_trunk_tz_id] < -0.24
        trunk_list_condition = (qpos[q_trunk_list_id] < -0.2793) or (
            qpos[q_trunk_list_id] > 0.2793
        )
        trunk_tilt_condition = (qpos[q_trunk_tilt_id] < -0.192) or (
            qpos[q_trunk_tilt_id] > 0.192
        )

        trunk_condition = (
            trunk_list_condition or trunk_tilt_condition or trunk_height_condition
        )

        return trunk_condition

    def _get_obs(self):

        position = self.data.qpos.flat.copy()[
            2:
        ]  # +2 avoids x-y position of trunk in the observation
        velocity = self.data.qvel.flat.copy()

        obs = np.concatenate((position, velocity, self.desired_velocity, np.array([self.phase])))

        return obs

    def reset_model(self):

        self.step_ = 0
        self.phase = 0

        if self.mode == "hard":
            random_vector = np.random.rand(2)
            norm = np.linalg.norm(random_vector)
            self.desired_velocity = (random_vector / norm) * 0.23 * self.target_speed_scale

        rand_idx = np.random.choice(self.init_samples.shape[0])
        init_states = self.init_samples[rand_idx]
        init_qpos = init_states[:18]
        init_qvel = init_states[18:]

        self.set_state(init_qpos, init_qvel)

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
