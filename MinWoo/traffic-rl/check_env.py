from stable_baselines3.common.env_checker import check_env
from env.intersection_env import IntersectionEnv


if __name__ == "__main__":
    env = IntersectionEnv()
    try:
        check_env(env, warn=True)
        print("Stable-Baselines3 check_env: PASS")
    finally:
        env.close()
