from agent import Agent
from snake_env import SnakeEnv


env = SnakeEnv()

state_size = env.state_size
action_size = 3

agent = Agent(state_size, action_size)


episodes = 1000

for episode in range(episodes):

    state = env.reset()

    done = False
    total_reward = 0

    while not done:

        action = agent.choose_action(state)

        next_state, reward, done = env.step(action)

        agent.train_step(state, action, reward, next_state, done)

        state = next_state

        total_reward += reward

    print("Episode:", episode, "Reward:", total_reward)