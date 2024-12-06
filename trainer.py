from env import GlucoseRegulationEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

env = GlucoseRegulationEnv()
check_env(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

model.save("ppo_glucose_regulation")#