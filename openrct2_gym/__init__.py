from gymnasium.envs.registration import register
import gymnasium as gym
from openrct2_gym.envs.wrappers import OpenRCT2Wrapper

register(
    id='OpenRCT2-v0',
    entry_point='openrct2_gym.envs:OpenRCT2Env',
)

def make_openrct2_env(render_mode=None, **kwargs):
    """
    Helper function to create an OpenRCT2 environment with proper wrapper.
    This ensures custom methods are accessible.
    """
    env = gym.make('OpenRCT2-v0', render_mode=render_mode, **kwargs)
    env = OpenRCT2Wrapper(env)
    return env
