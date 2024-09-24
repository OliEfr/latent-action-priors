import numpy as np
from pathlib import Path

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation
from torch import Value

DEFAULT_CAMERA_CONFIG = {"distance": 4.0, "elevation": -15.0}


class multiUnitreeA1ConnectedCustom(MujocoEnv, utils.EzPickle):
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
        n_unitreeA1=3,  # number of agents
        latent_action_space_dim: bool = False,
        mode: str = "easy",
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            n_unitreeA1,
            **kwargs,
        )
        
        raise ValueError("ToDo: implement random target sampling. Update XML with target indicator.")

        self.latent_action_space_dim = latent_action_space_dim

        self.n_unitreeA1 = n_unitreeA1

        self.mode = mode

        project_dir = Path(__file__).parent.parent.parent
        self.xml_path = str(
            project_dir
            / f"loco-mujoco/loco_mujoco/environments/data/quadrupeds/unitree_a1_torque_{self.n_unitreeA1}_coupled.xml"
        )

        obs_shape = (
            5 + 6 + 6 * self.n_unitreeA1 + 24 * self.n_unitreeA1 + 2
        )  # 5 (q) + 6 (dq) for connector, 3 (q + dq) rot joints for connecting unitreeA1 with connector, 12 (q) + 12 (dq) for each unitreeA1, 2 for self.desired_velocity
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        if mode == "easy":
            self.desired_velocity = np.array([0.23, 0.0])
        elif mode == "hard":
            random_vector = np.random.rand(2)
            norm = np.linalg.norm(random_vector)
            self.desired_velocity = random_vector / norm * 0.23
        else:
            raise ValueError("mode must be either 'easy' or 'hard'")
        self.prev_xpos = np.zeros((self.n_unitreeA1, 2))
        self.target_pos = np.array([1.5, 0.5])

        # initial states to sample from; taken from loco_mujoco
        self.init_samples = np.load("./unitree_a1_traj_init_samples.npy")

        self.start_episode = True

        MujocoEnv.__init__(
            self,
            self.xml_path,
            10,  # frame_skip
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            width=1920,
            height=1080,
            camera_name="track___1",
            **kwargs,
        )

    def step(self, action):

        # https://github.com/google-deepmind/mujoco/issues/1593
        # Upon detecting divergence, MuJoCo automatically resets the simulation and writes into mjData.warning. [...]. So your options are:
        # Check if data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0.
        # Check if mjData.time is 0, not at the start of the episode.

        # detect sim divergence
        divergence_detected = False
        if (
            not self.start_episode
            and self.data.time < self.model.opt.timestep * self.frame_skip
        ):
            divergence_detected = True
            print("Divergence detected. Terminating episode.")

        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()
        reached_goal = self._reached_goal()

        terminated = self._has_fallen() or divergence_detected or reached_goal

        reward = self._get_reward()

        info = {"task_reward": reward, "reached_goal": float(reached_goal)}

        # don't log this additional reward term
        if reached_goal:
            reward += 500

        self.start_episode = False

        return obs, reward, terminated, False, info

    def _reached_goal(self):
        xpos = self.data.xpos.copy()

        xpos_connector_id = self.model.body(f"connector").id  # x-y-z
        xpos_connector = xpos[xpos_connector_id][0:2]

        return np.linalg.norm(xpos_connector - self.target_pos) <= 0.1

    def _get_reward(self):
        # reward is difference of current position from target position

        # curr_pos = self.data.qpos.flat.copy()[:2]
        # reward = np.exp(3.0 * -np.linalg.norm(curr_pos - self.target_pos))
        xpos = self.data.xpos.copy()

        xpos_connector_id = self.model.body(f"connector").id  # x-y-z
        xpos_connector = xpos[xpos_connector_id][0:2]
        reward = np.exp(-np.linalg.norm(xpos_connector - self.target_pos))

        # reward is mean difference of agents from target velocity
        # for i in range(1, self.n_unitreeA1 + 1):
        #     xpos_trunk_tz_id = self.model.body(
        #         f"trunk___{i}"
        #     ).id  # x-y-z coordinates for every body
        #     xpos_trunk_i = xpos[xpos_trunk_tz_id][0:2]
        #     vel_trunk_i = (xpos_trunk_i - self.prev_xpos[i - 1]) / self.dt
        #     self.prev_xpos[i - 1] = xpos_trunk_i
        #     reward.append(
        #         np.exp(-5.0 * np.linalg.norm(vel_trunk_i - self.desired_velocity))
        #     )

        # # curr_velocity_xy = self.data.qvel.copy()[0:1]

        # # reward = np.exp(-5.0 * np.linalg.norm(curr_velocity_xy - self.desired_velocity))

        # reward = np.mean(reward)

        return reward

    def _has_fallen(self):

        trunks_tz = []
        # trunks_tx = []

        xpos = self.data.xpos.copy()
        xmat = self.data.xmat.copy()

        all_trunk_conditions = []
        for i in range(1, self.n_unitreeA1 + 1):
            # access termination conditions via body
            xpos_trunk_tz_id = self.model.body(
                f"trunk___{i}"
            ).id  # x-y-z coordinates for every body
            trunk_height_condition = (
                xpos[xpos_trunk_tz_id][2] < 0.19
            )  # corresponds to trunk_height_condition from original loco_mujoco env
            # trunks_tx.append(xpos[xpos_trunk_tz_id][0])
            trunks_tz.append(xpos[xpos_trunk_tz_id][2])

            xmat_i = xmat[xpos_trunk_tz_id].reshape(3, 3)
            angles = Rotation.from_matrix(xmat_i).as_euler("XYZ", degrees=False)[:2]

            trunk_list_condition = (angles[0] < -0.2793) or (angles[0] > 0.2793)
            trunk_tilt_condition = (angles[1] < -0.192) or (angles[1] > 0.192)

            trunk_condition = (
                trunk_list_condition or trunk_tilt_condition or trunk_height_condition
            )

            all_trunk_conditions.append(trunk_condition)

        # trunk_dz_condition
        all_trunk_conditions.append(np.ptp(trunks_tz) > 0.1)
        # all_trunk_conditions.append(np.ptp(trunks_tx) > 0.1)

        return any(all_trunk_conditions)  # use any() or all()

    def _get_obs(self):

        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        # distance = position[:2] - self.target_pos # vectorial x-y difference

        xpos = self.data.xpos.copy()

        xpos_connector_id = self.model.body(f"connector").id  # x-y-z
        xpos_connector = xpos[xpos_connector_id][0:2]

        return np.concatenate(
            (position[2:], velocity, self.target_pos - xpos_connector)
        )

    def reset_model(self):

        self.start_episode = True

        if self.mode == "hard":
            random_vector = np.random.rand(2)
            norm = np.linalg.norm(random_vector)
            self.desired_velocity = random_vector / norm * 0.23

        # joints_per_unitreeA1 = 18 # = self.model.njnt / self.n_unitreeA1
        qpos = []
        qvel = []

        # connector joint
        qpos += self.init_qpos.tolist()[:7]
        qvel += self.init_qvel.tolist()[:6]

        # use same init states for all unitreeA1, else the connector-bar will not be perfectly horizontal
        rand_idx = np.random.choice(self.init_samples.shape[0])
        init_states = self.init_samples[rand_idx]
        qpos[2] += init_states[
            2
        ]  # set z-height of connector (and thereby of the trunks).

        for _ in range(1, self.n_unitreeA1 + 1):
            # states of joints from connector to unitreeA1
            qpos += [0, 0, 0]
            qvel += [0, 0, 0]

            # dont use first 6, as they correspond to unitreeA1 trunk, which is not used in this env
            init_qpos = init_states[6:18].tolist()
            init_qvel = init_states[18 + 6 :].tolist()

            qpos += init_qpos
            qvel += init_qvel

        qpos = np.array(qpos)
        qvel = np.array(qvel)

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


class multiUnitreeA1ConnectedHasExpertDataCustom(MujocoEnv, utils.EzPickle):
    """
    Gymnasium environment for the Unitree A1 quadruped robot that is able to spawn multiple agents.

    """

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
        n_unitreeA1=3,  # number of agents
        latent_action_space_dim: bool = False,
        mode: str = "easy",
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            n_unitreeA1,
            **kwargs,
        )

        self.latent_action_space_dim = latent_action_space_dim

        self.n_unitreeA1 = n_unitreeA1
        if n_unitreeA1 == 3:
            raise ValueError("Update xml with target indicator.")

        self.mode = mode

        project_dir = Path(__file__).parent.parent.parent
        self.xml_path = str(
            project_dir
            / f"loco-mujoco/loco_mujoco/environments/data/quadrupeds/unitree_a1_torque_{self.n_unitreeA1}_coupled.xml"
        )

        obs_shape = (
            5 + 6 + 6 * self.n_unitreeA1 + 24 * self.n_unitreeA1 + 2 + 1
        )  # 5 (q) + 6 (dq) for connector, 3 (q + dq) rot joints for connecting unitreeA1 with connector, 12 (q) + 12 (dq) for each unitreeA1, 2 for self.desired_velocity, 1 for phase
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        # self.target_pos = np.array([1, 0.5])
        if mode == "easy":
            self.desired_velocity = np.array([0.23, 0.0])
        elif mode == "hard":
            random_vector = np.random.rand(2)
            norm = np.linalg.norm(random_vector)
            self.desired_velocity = random_vector / norm * 0.23
        else:
            raise ValueError("mode must be either 'easy' or 'hard'")
        self.prev_xpos = np.zeros((self.n_unitreeA1, 2))
        self.target_pos = np.array([1.0, 0.5])

        # initial states to sample from; taken from loco_mujoco
        self.init_samples = np.load("./unitree_a1_traj_init_samples.npy")

        self.start_episode = True

        MujocoEnv.__init__(
            self,
            self.xml_path,
            10,  # frame_skip
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            width=1920,
            height=1080,
            camera_name="track___1",
            **kwargs,
        )

        expert_data = np.load(
            "recorded_experts/loco_mujoco/loco_mujoco_a1.simple_latent_6_nonLin/UnitreeA1.simple.perfect.npy",
            allow_pickle=True,
        ).item()
        self.expert_obs_cycle = expert_data["recorded_obs_cycle"].copy()

        self.step_ = 0
        self.phase = 0

    def imitating_joint_pos_reward(self):
        # perform check for all unitreeA1s
        joint_pos = self.data.qpos[7:].copy()  # dont consider free-type connector joint
        assert (
            joint_pos.shape[0] == self.n_unitreeA1 * 15
        ), "Should have 15 joints per unitreeA1"
        
        joint_pos_ref = self.expert_obs_cycle[self.step_ % len(self.expert_obs_cycle)][
            1:16
        ]  # dont consider z-height joint as this DOF is given to connector

        joint_pos_ref = np.tile(joint_pos_ref, self.n_unitreeA1)

        return np.exp(-np.linalg.norm(joint_pos_ref - joint_pos))

    def step(self, action):

        # https://github.com/google-deepmind/mujoco/issues/1593
        # Upon detecting divergence, MuJoCo automatically resets the simulation and writes into mjData.warning. [...]. So your options are:
        # Check if data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0.
        # Check if mjData.time is 0, not at the start of the episode.

        # detect sim divergence
        divergence_detected = False
        if (
            not self.start_episode
            and self.data.time < self.model.opt.timestep * self.frame_skip
        ):
            divergence_detected = True
            print("Divergence detected. Terminating episode.")

        self.do_simulation(action, self.frame_skip)
        self.step_ += 1
        self.phase = self.step_ % len(self.expert_obs_cycle)

        obs = self._get_obs()
        reached_goal = self._reached_goal()

        terminated = self._has_fallen() or divergence_detected or reached_goal
        task_reward = self._get_reward()

        style_reward = self.imitating_joint_pos_reward()
        
        info = {
            "task_reward": task_reward,
            "style_reward": style_reward,
            "joint_pos_reward": style_reward,
            "reached_goal": float(reached_goal),
        }
        
        # don't log this additional reward term
        if reached_goal:
            task_reward += 500

        reward = 0.67 * task_reward + 0.33 * style_reward

        self.start_episode = False

        return obs, reward, terminated, False, info

    def _reached_goal(self):
        xpos = self.data.xpos.copy()

        xpos_connector_id = self.model.body(f"connector").id  # x-y-z
        xpos_connector = xpos[xpos_connector_id][0:2]

        return np.linalg.norm(xpos_connector - self.target_pos) <= 0.1

    def _get_reward(self):
        # reward is difference of current position from target position

        # curr_pos = self.data.qpos.flat.copy()[:2]
        # reward = np.exp(3.0 * -np.linalg.norm(curr_pos - self.target_pos))
        xpos = self.data.xpos.copy()

        xpos_connector_id = self.model.body(f"connector").id  # x-y-z
        xpos_connector = xpos[xpos_connector_id][0:2]
        reward = np.exp(-np.linalg.norm(xpos_connector - self.target_pos))

        # reward is mean difference of agents from target velocity
        # for i in range(1, self.n_unitreeA1 + 1):
        #     xpos_trunk_tz_id = self.model.body(
        #         f"trunk___{i}"
        #     ).id  # x-y-z coordinates for every body
        #     xpos_trunk_i = xpos[xpos_trunk_tz_id][0:2]
        #     vel_trunk_i = (xpos_trunk_i - self.prev_xpos[i - 1]) / self.dt
        #     self.prev_xpos[i - 1] = xpos_trunk_i
        #     reward.append(
        #         np.exp(-5.0 * np.linalg.norm(vel_trunk_i - self.desired_velocity))
        #     )

        # # curr_velocity_xy = self.data.qvel.copy()[0:1]

        # # reward = np.exp(-5.0 * np.linalg.norm(curr_velocity_xy - self.desired_velocity))

        # reward = np.mean(reward)

        return reward

    def _has_fallen(self):

        trunks_tz = []
        # trunks_tx = []

        xpos = self.data.xpos.copy()
        xmat = self.data.xmat.copy()

        all_trunk_conditions = []
        for i in range(1, self.n_unitreeA1 + 1):
            # access termination conditions via body
            xpos_trunk_tz_id = self.model.body(
                f"trunk___{i}"
            ).id  # x-y-z coordinates for every body
            trunk_height_condition = (
                xpos[xpos_trunk_tz_id][2] < 0.19
            )  # corresponds to trunk_height_condition from original loco_mujoco env
            # trunks_tx.append(xpos[xpos_trunk_tz_id][0])
            trunks_tz.append(xpos[xpos_trunk_tz_id][2])

            xmat_i = xmat[xpos_trunk_tz_id].reshape(3, 3)
            angles = Rotation.from_matrix(xmat_i).as_euler("XYZ", degrees=False)[:2]

            trunk_list_condition = (angles[0] < -0.2793) or (angles[0] > 0.2793)
            trunk_tilt_condition = (angles[1] < -0.192) or (angles[1] > 0.192)

            trunk_condition = (
                trunk_list_condition or trunk_tilt_condition or trunk_height_condition
            )

            all_trunk_conditions.append(trunk_condition)

        # trunk_dz_condition
        all_trunk_conditions.append(np.ptp(trunks_tz) > 0.1)
        # all_trunk_conditions.append(np.ptp(trunks_tx) > 0.1)

        return any(all_trunk_conditions)  # use any() or all()

    def _get_obs(self):

        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        # distance = position[:2] - self.target_pos # vectorial x-y difference

        xpos = self.data.xpos.copy()

        xpos_connector_id = self.model.body(f"connector").id  # x-y-z
        xpos_connector = xpos[xpos_connector_id][0:2]

        return np.concatenate(
            (
                position[2:],
                velocity,
                self.target_pos - xpos_connector,
                np.array([self.phase]),
            )
        )

    def reset_model(self):
        
        # sample random target point in 45° angle around x axis and 1m radius
        theta = np.random.uniform(-np.pi/4, np.pi/4)
        x = np.cos(theta)
        y = np.sin(theta) + 0.5 # 0.5 is center of connector in y-direction
        self.target_pos = np.array([x, y])
        
        # update target visual
        self.model.site(f"target").pos[0:2] = self.target_pos

        self.start_episode = True
        self.step_ = 0
        self.phase = 0

        if self.mode == "hard":
            random_vector = np.random.rand(2)
            norm = np.linalg.norm(random_vector)
            self.desired_velocity = random_vector / norm * 0.23

        # joints_per_unitreeA1 = 18 # = self.model.njnt / self.n_unitreeA1
        qpos = []
        qvel = []

        # connector joint
        qpos += self.init_qpos.tolist()[:7]
        qvel += self.init_qvel.tolist()[:6]

        # use same init states for all unitreeA1, else the connector-bar will not be perfectly horizontal
        rand_idx = np.random.choice(self.init_samples.shape[0])
        init_states = self.init_samples[rand_idx]
        qpos[2] += init_states[
            2
        ]  # set z-height of connector (and thereby of the trunks).

        for _ in range(1, self.n_unitreeA1 + 1):
            # states of joints from connector to unitreeA1
            qpos += [0, 0, 0]
            qvel += [0, 0, 0]

            # dont use first 6, as they correspond to unitreeA1 trunk, which is not used in this env
            init_qpos = init_states[6:18].tolist()
            init_qvel = init_states[18 + 6 :].tolist()

            qpos += init_qpos
            qvel += init_qvel

        qpos = np.array(qpos)
        qvel = np.array(qvel)

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
