import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

# Training Environment 
def create_env():
    env = gym.make("CarRacing-v3", render_mode="rgb_array")  # render_mode 추가
    env = Monitor(env)
    return env

env = DummyVecEnv([create_env])
env = VecTransposeImage(env)

eval_env = DummyVecEnv([create_env])
eval_env = VecTransposeImage(eval_env)

model = SAC(
    "CnnPolicy",
    env,
    verbose=1,
    buffer_size=100000,
    learning_rate=3e-4,
    batch_size=256,
    train_freq=1,
    gradient_steps=1,
    learning_starts=1000,
    gamma=0.99,
    tau=0.005,
    ent_coef="auto",
    tensorboard_log="./sac_carracing_tensorboard/"  # TensorBoard 로그 경로
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./sac_carracing_best_model/",
    log_path="./sac_carracing_eval_logs/",
    eval_freq=10000,
    n_eval_episodes=5,
    deterministic=True, 
    render=False        
)

model.learn(total_timesteps=500000, callback=eval_callback)

model.save("sac_carracing_model")

loaded_model = SAC.load("sac_carracing_model")

obs = env.reset()

for _ in range(1000):
    action, _states = loaded_model.predict(obs, deterministic=True)  # Deterministic Action
    obs, reward, done, info = env.step(action)
    env.render() 
    if done:
        obs = env.reset()

env.close()
