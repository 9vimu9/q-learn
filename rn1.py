import gym
import numpy as np

env = gym.make("MountainCar-v0")
state = env.reset()
print(env.observation_space.high)
print(env.observation_space.low)

DISCRETE_OS_SIZE = [20, 20]

discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
print(discrete_os_win_size)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
# So, this is a 20x20x3 shape, which has initialized random Q values(max 0 min -2) for us. The 20 x 20 bit 
# is every combination of the bucket slices of all possible states. The x3 bit is for every 
# possible action we could take.
print(q_table.shape)# (20, 20, 3)
#argmax returns the position of the largest value. max returns the largest value.
done = False
while not done:
    action = 0  
    new_state, reward, done, _ = env.step(action)
    print(reward, new_state)
    env.render()