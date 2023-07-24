from stable_baselines3.common.env_checker import check_env
from wildfire_env_v3 import WildFireEnv


def check_env():
    env = WildFireEnv()
    # It will check your custom environment and output additional warnings if needed
    check_env(env)
    print('Passing')

if __name__ == "__main__":
    check_env()
