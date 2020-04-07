import gym
import numpy as np

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1

DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 2000


'''
Okay, great, but it doesn't appear that the model is learning. Why not?

So the Q-Table is initialized....randomly. Then, every step has the agent getting rewarded with a -1. 
The only time the agent gets rewarded with, well, nothing (a 0)... is if they reach the objective. 
We need the agent to reach the objective some time. If they just reach it once, they will be more 
likely to reach it again as the reward back propagates. As they get more likely to reach it, they'll 
each it again and again...and boom, they will have learned. But, how do we get to reach it the first 
time?!

Epsilon!

Or, as regular people might call it: random moves.

As an Agent learns an environment, it moves from "exploration" to "exploitation." Right now, our model 
is greedy and exploiting for max Q values always...but these Q values are worthless right now. We need 
the agent to explore!

For this, we'll add the following values:
'''
# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
#convert continues state values to discrete values . this will reduce our q_table size

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
#create qtable with random valiues(max 0 min -2)
#size is 20x20x3

def get_discrete_state(state):# function to convert continues to discrete
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table


for episode in range(EPISODES):

    discrete_state = get_discrete_state(env.reset())#set inital state to discrete_state

    done = False

    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False

    while not done :

        if np.random.random() > epsilon/2:
            # Get action from Q table

            q_values = q_table[discrete_state] 
            #get relevent q values of discrete_state from q table 
            #(in q table we have q values for every action of every possible state )
            action = np.argmax(q_values)
            #we  get maximum q value's action as action of a state so we use argmax to get index of max
            #actions ==> left ,still, right 
            #numerical value for each action ==> 0,1,2
            # we store q values of each states's actions in order of actions numerical value
            
            #example
            #state == (5,10) q values => -1.244535 for left -1.46647 for stop 0 for right [-1.233535,-1.466747,0]
            # so we can use argmax to get action of max q value
            
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

     

        new_state, reward, done, _ = env.step(action)
        #do a step and get new state and reward
        #done will return true if we finish task

        new_discrete_state = get_discrete_state(new_state)

        if render :
            env.render()

        # If simulation did not end yet after last step - update Q table
        if not done:

            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q


        # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
        elif new_state[0] >= env.goal_position:
            #q_table[discrete_state + (action,)] = reward
            q_table[discrete_state + (action,)] = 0
            print(f"we made it on episode: {episode}")

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()