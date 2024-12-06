import gymnasium as gym  
import numpy as np
from gymnasium import spaces
import random

class GlucoseRegulationEnv(gym.Env): 
    def __init__(self):
        super(GlucoseRegulationEnv, self).__init__()
        
        
        self.target_glucose_min = 70  
        self.target_glucose_max = 100 
        self.init_glucose = 150       
        self.glucose = self.init_glucose
        
        
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        
        
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0]), 
                                            high=np.array([500, 10, 10, 100, 24]), dtype=np.float32)
        
        self.insulin_prev = 0
        self.glucagon_prev = 0
        self.time_of_day = 0
        self.time_step = 0
        self.carb_intake = 0
        self.done = False

        
        self.rng = np.random.default_rng()

    def seed(self, seed=None):
        
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        
        self.glucose = self.init_glucose
        self.insulin_prev = 0
        self.glucagon_prev = 0
        self.time_of_day = 0
        self.time_step = 0
        self.carb_intake = self.rng.uniform(0, 100)  
        self.done = False
        
        return np.array([self.glucose, self.insulin_prev, self.glucagon_prev, self.carb_intake, self.time_of_day], dtype=np.float32), {}

    def step(self, action):
        insulin_dose = (action[0] + 1) * 5 
        glucagon_dose = (action[1] + 1) * 5 
        
        
        self.glucose += self.simulate_glucose_dynamics(insulin_dose, glucagon_dose)
        
       
        if self.time_step % 24 in [8, 12, 18]:  
            self.carb_intake = self.rng.uniform(50, 500)
        else:
            self.carb_intake = 0
        
        self.glucose += self.carb_intake * 0.1
        
        
        self.time_of_day = (self.time_step % 96) / 4  
        self.time_step += 1
        
        
        reward = self.calculate_reward()
        
        
        self.insulin_prev = insulin_dose
        self.glucagon_prev = glucagon_dose
        
        
        terminated = self.time_step >= 96
        truncated = False  
        
        
        obs = np.array([self.glucose, self.insulin_prev, self.glucagon_prev, self.carb_intake, self.time_of_day], dtype=np.float32)
        return obs, reward, terminated, truncated, {}

    def simulate_glucose_dynamics(self, insulin_dose, glucagon_dose):
        
        insulin_effect = -2.0 * insulin_dose
        
        
        glucagon_effect = 1.0 * glucagon_dose
        
        
        return insulin_effect + glucagon_effect
    
    def calculate_reward(self):
        
        if self.target_glucose_min <= self.glucose <= self.target_glucose_max:
            return 1.0  
        else:
            
            return -abs(self.glucose - 105) / 10.0