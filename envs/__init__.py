from gymnasium.envs.registration import register

register(
    id="HumanoidCustom-v4",
    entry_point="envs.envs:HumanoidCustom_v4",
    max_episode_steps=1000,
)

register(
    id="HumanoidHasExpertDataCustom-v4",
    entry_point="envs.envs:HumanoidHasExpertDataCustom_v4",
    max_episode_steps=1000,
)

register(
    id="AntCustom-v4",
    entry_point="envs.envs:AntCustom_v4",
    max_episode_steps=1000,
)

register(
    id="AntHasExpertDataCustom-v4",
    entry_point="envs.envs:AntHasExpertDataCustom_v4",
    max_episode_steps=1000,
)

register(
    id="HalfCheetahCustom-v4",
    entry_point="envs.envs:HalfCheetahCustom_v4",
    max_episode_steps=1000,
)

register(
    id="HalfCheetahHasExpertDataCustom-v4",
    entry_point="envs.envs:HalfCheetahHasExpertDataCustom_v4",
    max_episode_steps=1000,
)


register(
    id="multiUnitreeA1ConnectedCustom-v0",
    entry_point="envs.envs:multiUnitreeA1ConnectedCustom",
    max_episode_steps=1000,
)

register(
    id="UnitreeA1Custom-v0",
    entry_point="envs.envs:UnitreeA1Custom",
    max_episode_steps=1000,
)

register(
    id="UnitreeA1HasExpertDataCustom-v0",
    entry_point="envs.envs:UnitreeA1HasExpertDataCustom",
    max_episode_steps=1000,
)


register(
    id="multiUnitreeA1ConnectedHasExpertDataCustom-v0",
    entry_point="envs.envs:multiUnitreeA1ConnectedHasExpertDataCustom",
    max_episode_steps=1000,
)







