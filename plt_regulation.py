import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from env import GlucoseRegulationEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import plotly.graph_objects as go


env = GlucoseRegulationEnv()
check_env(env)
model = PPO.load("ppo_glucose_regulation", env=env) 



def simulate_and_plot(model, env, time_steps=96):
   
    glucose_levels = []
    insulin_levels = []
    glucagon_levels = []
    times = []

    obs, _ = env.reset()
    done = False
    step_count = 0

    while not done and step_count < time_steps:
        
        action, _states = model.predict(obs)
        
       
        obs, reward, terminated, truncated, info = env.step(action)
        
        
        glucose = obs[0]
        insulin = obs[1]
        glucagon = obs[2]
        time_of_day = obs[4] * 4  
        
        glucose_levels.append(glucose)
        insulin_levels.append(insulin)
        glucagon_levels.append(glucagon)
        times.append((time_of_day*15)/60)
        
        
        if step_count % 12 == 0:
            print(f"Time: {time_of_day:.2f} hours")
            print(f"  Glucose: {glucose:.2f} mg/dL")
            print(f"  Insulin: {insulin:.2f} units")
            print(f"  Glucagon: {glucagon:.2f} units\n")
        
        done = terminated or truncated
        step_count += 1
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, glucose_levels, label="Glucose Level (mg/dL)", color="b")
    plt.axhline(y=140, color="g", linestyle="--", label="Upper Target (140 mg/dL)")
    plt.axhline(y=70, color="r", linestyle="--", label="Lower Target (70 mg/dL)")
    plt.xlabel("Time of Day (hours)")
    plt.ylabel("Glucose Level (mg/dL)")
    plt.title("Agent Performance in Maintaining Blood Glucose Levels")
    plt.legend()
    plt.grid(True)
    plt.show()
def simulate_and_plot1(model, env, time_steps= 100):  

    glucose_levels = []
    insulin_levels = []
    glucagon_levels = []
    times = []

    obs, _ = env.reset()
    done = False
    step_count = 0

    while not done and step_count < time_steps:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        glucose = obs[0]
        insulin = obs[1]
        glucagon = obs[2]
        time_of_day = obs[4] * 4 

        glucose_levels.append(glucose)
        insulin_levels.append(insulin)
        glucagon_levels.append(glucagon)
        times.append(time_of_day)

        done = terminated or truncated
        step_count += 1

    
    fig, ax = plt.subplots()
    line_glucose, = ax.plot(times, glucose_levels, label="Glucose (mg/dL)")
    line_insulin, = ax.plot(times, insulin_levels, label="Insulin (units)")
    line_glucagon, = ax.plot(times, glucagon_levels, label="Glucagon (units)")
    ax.set_xlabel("Time of Day (hours)")
    ax.set_ylabel("Level")
    ax.set_title("Blood Glucose Regulation Simulation")
    ax.legend()

    def animate(i):
        line_glucose.set_data(times[:i+1], glucose_levels[:i+1])
        line_insulin.set_data(times[:i+1], insulin_levels[:i+1])
        line_glucagon.set_data(times[:i+1], glucagon_levels[:i+1])
        return line_glucose, line_insulin, line_glucagon

    ani = FuncAnimation(fig, animate, frames=len(times), interval=50, blit=True)
    plt.show()
def simulate_and_plot_with_plotly(model, env, time_steps=96):
    # Data lists
    glucose_levels = []
    insulin_levels = []
    glucagon_levels = []
    times = []

    # Reset environment
    obs, _ = env.reset()
    done = False
    step_count = 0

    while not done and step_count < time_steps:
        # Predict action from the model
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        # Extract relevant data
        glucose = obs[0]
        insulin = obs[1]
        glucagon = obs[2]
        time_of_day = obs[4] * 4  

        # Append data to lists
        glucose_levels.append(glucose)
        insulin_levels.append(insulin)
        glucagon_levels.append(glucagon)
        times.append(time_of_day)

        # Check termination
        done = terminated or truncated
        step_count += 1

    # Create Plotly figure
    fig = go.Figure()

    # Add glucose levels
    fig.add_trace(go.Scatter(
        x=times, y=glucose_levels, mode='lines', name='Glucose (mg/dL)',
        line=dict(color='blue')
    ))

    # Add insulin levels
    fig.add_trace(go.Scatter(
        x=times, y=insulin_levels, mode='lines', name='Insulin (units)',
        line=dict(color='orange')
    ))

    fig.add_trace(go.Scatter(
        x=times, y=glucagon_levels, mode='lines', name='Glucagon (units)',
        line=dict(color='green')
    ))

    fig.add_trace(go.Scatter(
        x=times, y=[100] * len(times), mode='lines', name='Upper Target (140 mg/dL)',
        line=dict(color='green', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=times, y=[70] * len(times), mode='lines', name='Lower Target (70 mg/dL)',
        line=dict(color='red', dash='dash')
    ))

    # Customize layout
    fig.update_layout(
        title="Blood Glucose Regulation Simulation",
        xaxis_title="Time of Day (hours)",
        yaxis_title="Level",
        legend_title="Metrics",
        template="plotly_white"
    )

    # Show the plot
    fig.show()

# Run the simulation and plot with Plotly
simulate_and_plot_with_plotly(model, env)

## simulate_and_plot1(model, env)
