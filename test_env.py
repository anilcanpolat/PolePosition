import gymnasium as gym
import numpy as np


class Tester:

    def __init__(self, env_id, obs_type):
        self.env_id = env_id
        self.obs_type = obs_type
        self.fails = []

    def fail(self, msg):
        self.fails.append(msg)

    def test(self, x, msg):
        if x:
            return True
        else:
            self.fail(msg)
            return False
    
    def check_all(self, *args):
        if len(args) % 2 != 0:
            raise ValueError()
        
        result = True
        for i in range(0, len(args), 2):
            if not self.test(args[i](), args[i + 1]):
                result = False
                break
        return result

    def finish(self):
        print('==================================================================')
        print(f'Testing environment with id "{self.env_id}"')
        print(f'Expected observation type: "{self.obs_type}"')
        if len(self.fails) == 0:
            print('All tests succeeded!')
        else:
            for fail in self.fails:
                print(f'Fail: {fail}')
        print('==================================================================\n')
        return len(self.fails) == 0


def test_env(env_id, obs_type):
    if obs_type not in ('pixels', 'features'):
        raise ValueError(obs_type)

    tester = Tester(env_id, obs_type)

    try:
        env = gym.make(env_id, render_mode='rgb_array')
    except gym.error.NameNotFound:
        tester.fail(f'Environment {env_id} is not registered')
        return tester.finish()
    except gym.error.Error as e:
        tester.fail(f'Failed to create environment: {e}')
        return tester.finish()
    except:
        tester.fail(f'Failed to create environment')
        return tester.finish()

    if not tester.test(isinstance(env, gym.Env), 'Environment is not a subclass of gym.Env'):
        return tester.finish()

    act_space_ok = False
    if tester.test(hasattr(env, 'action_space'), 'Environment has no action space'):
        act_space = env.action_space
        act_space_ok = tester.check_all(
            lambda: isinstance(act_space, gym.spaces.Discrete), 'action_space is not a subclass of gym.spaces.Discrete',
            lambda: act_space.n <= 8, 'action_space contains more than 8 actions',
        )

    obs_space_ok = False
    obs_space_flattened = False
    if tester.test(hasattr(env, 'observation_space'), 'Environment has no observation space'):
        obs_space = env.observation_space
        if obs_type == 'pixels':
            obs_space_ok = tester.check_all(
                lambda: isinstance(obs_space, gym.spaces.Box), 'observation_space is not a subclass of gym.spaces.Box',
                lambda: obs_space.shape in ((64, 64, 3), (84, 84, 3)), 'observation_space is not of shape (64, 64, 3) or (84, 84, 3), i.e. (Height x Width x RGB channels)',
                lambda: obs_space.dtype == np.uint8, 'observation_space does not have data type np.uint8',
            )
        elif obs_type == 'features':
            try:
                flat_obs_space = gym.spaces.utils.flatten_space(obs_space)
                if tester.test(isinstance(flat_obs_space, gym.spaces.Box), 'observation_space cannot be flattened to a gym.spaces.Box'):
                    obs_space_ok = tester.check_all(
                        lambda: len(flat_obs_space.shape) == 1, f'observation_space can be flattened to a gym.spaces.Box, but the flattened shape is not 1-dimensional, got {flat_obs_space.shape}',
                        lambda: flat_obs_space.shape[0] <= 2048, f'observation_space is too large, got {flat_obs_space.shape[0]} values but should not be more than 2048',
                        lambda: np.issubdtype(flat_obs_space.dtype, np.floating), 'observation_space does not have data type np.floating',
                    )
                obs_space_flattened = True
            except NotImplementedError:
                tester.fail('observation_space cannot be flattened to a gym.spaces.Box')
        else:
            raise ValueError()

    if obs_space_ok and act_space_ok:
        try:
            reset = env.reset()
        except:
            tester.fail('env.reset() throws an exception')
            return tester.finish()

        if tester.test(isinstance(reset, tuple) and len(reset) == 2, 'env.reset() does not return a tuple of (observation, info)'):
            obs, info = reset
            if tester.test(obs_space.contains(obs), 'observation returned by env.reset() does not belong to the observation space') and obs_space_flattened:
                # this should never fail?
                flat_obs = gym.spaces.utils.flatten(obs_space, obs)
                tester.test(flat_obs_space.contains(flat_obs), 'observation returned by env.reset() does not belong to the flattened observation space')

        action = act_space.sample()

        try:
            step = env.step(action)
        except:
            tester.fail('env.step() throws an exception')
            return tester.finish()

        if tester.test(isinstance(step, tuple) and len(step) == 5, 'env.step() does not return a tuple of (observation, reward, terminated, truncated, info)'):
            obs, reward, terminated, truncated, info = step
            if tester.test(obs_space.contains(obs), 'observation returned by env.step() does not belong to the observation space') and obs_space_flattened:
                # this should never fail?
                flat_obs = gym.spaces.utils.flatten(obs_space, obs)
                tester.test(flat_obs_space.contains(flat_obs), 'observation returned by env.step() does not belong to the flattened observation space')
            tester.test(isinstance(reward, (float, np.floating)), 'reward returned by env.step() is not a float')
            tester.test(isinstance(terminated, bool), '`terminated` returned by env.step() is not a bool')
            tester.test(isinstance(truncated, bool), '`truncated` returned by env.step() is not a bool')

        rgb_array = env.render()
        tester.check_all(
            lambda: isinstance(rgb_array, np.ndarray), 'env.render() does not return a NumPy array, although render_mode is set to "rgb_array"',
            lambda: len(rgb_array.shape) == 3 and rgb_array.shape[-1] == 3, f'rgb_array returned by env.render() is not of shape (height, width, 3) (RGB channels), got {rgb_array.shape}',
        )

    return tester.finish()


def test_env_pixels(env_id):
    test_env(env_id, obs_type='pixels')


def test_env_features(env_id):
    test_env(env_id, obs_type='features')


if __name__ == '__main__':
    # Import your environment, so it is registered in gym
    
    from PolePosition import *  # Ensure this imports the __init__.py with the registrations

    # Replace by the correct environment ids
    test_env_features('PolePosition-features-v0')
    test_env_pixels('PolePosition-pixels-v0')