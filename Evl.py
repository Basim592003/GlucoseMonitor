from env import GlucoseRegulationEnv
from stable_baselines3 import PPO
#
env = GlucoseRegulationEnv()
model = PPO.load("ppo_glucose_regulation", env=env) 

def evaluate_agent(model, env, episodes=10):
    results = {
        'avg_glucose': [],
        'time_in_range': [],
        'hypo_events': [],
        'hyper_events': []
    }
    
    for episode in range(episodes):
        obs, _ = env.reset()  
        done = False
        total_glucose = 0
        time_in_range = 0
        hypo_events = 0
        hyper_events = 0
        time_step = 0
        
        while not done:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            glucose = obs[0]
            total_glucose += glucose
            
            
            if 70 <= glucose <= 140:
                time_in_range += 1
                
            
            if glucose < 70:
                hypo_events += 1
            elif glucose > 140:
                hyper_events += 1
            
            
            done = terminated or truncated
            time_step += 1
        
        avg_glucose = total_glucose / time_step
        results['avg_glucose'].append(avg_glucose)
        results['time_in_range'].append(time_in_range / time_step)
        results['hypo_events'].append(hypo_events)
        results['hyper_events'].append(hyper_events)
    
    return results


eval_results = evaluate_agent(model, env)  
print("Evaluation Results:", eval_results)
