import gymnasium as gym
import gymnasium_env
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

EPISODES = 1000
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
MAX_STEPS = 500

env = gym.make("SimpleMaze-v0")

grid_shape = env.unwrapped.grid_size
q_table = np.zeros((grid_shape[0], grid_shape[1], env.action_space.n))

rewards_per_episode = []

for episode in range(EPISODES):
    steps = 0
    state, _ = env.reset()
    # if (episode + 1) % 50 == 0:
    #     print(">>> Grid przed treningiem:")
    #     print(env.unwrapped.grid)

    done = False
    total_reward = 0
    recent_positions = []

    while not done and steps < MAX_STEPS:
        x, y = state

        if np.random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[x, y])

        next_state, reward, done, _, _ = env.step(action)

        recent_positions.append(tuple(next_state))
        if len(recent_positions) > 10:
            recent_positions.pop(0)
        if recent_positions.count(tuple(next_state)) > 4:
            reward -= 3

        nx, ny = next_state
        old_value = q_table[x, y, action]
        next_max = np.max(q_table[nx, ny])
        new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
        q_table[x, y, action] = new_value

        state = next_state
        total_reward += reward
        steps += 1

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    rewards_per_episode.append(total_reward)

    if (episode + 1) % 50 == 0:
        print(f"Epizod {episode + 1}, suma nagród: {total_reward:.2f}, epsilon: {EPSILON:.3f}")


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


rolling_mean = moving_average(rewards_per_episode, 50)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(rewards_per_episode, label='Nagroda (epizod)')
plt.title("Nagrody w trakcie treningu")
plt.xlabel("Epizod")
plt.ylabel("Nagroda")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(rolling_mean, label='Średnia krocząca (50)', color='orange')
plt.title("Wygładzona średnia nagród")
plt.xlabel("Epizod")
plt.ylabel("Średnia nagroda")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

env = gym.make("SimpleMaze-v0", render_mode="human")
state, _ = env.reset()
done = False

while not done:
    env.render()
    x, y = state
    action = np.argmax(q_table[x, y])
    state, _, done, _, _ = env.step(action)

env.close()
