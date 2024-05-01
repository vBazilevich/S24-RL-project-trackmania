"""
Custom Ad-hoc script for Trackmania driving.
This script conducts policy evaluation & trackmania environment interaction,
without training
"""
from tmrl import get_environment
from time import sleep
import numpy as np


# LIDAR observations are of shape: ((1,), (4, 19), (3,), (3,))
# representing: (speed, 4 last LIDARs, 2 previous actions)
# actions are [gas, break, steer], analog between -1.0 and +1.0
def model(obs):
    STEER_ALPHA = 0.3 # intensity of steering depending on distance
    STUCK_THRESH = 55 # threshold for going backwards if stuck

    deviation = obs[1].mean(0)

    # determine direction based on smallest distance to the border
    direction = np.argsort(deviation)[0]
    direction_to_steer = (-1 + (direction / 19) * 2) * STEER_ALPHA
    steer = min(max(direction_to_steer, -1.0), 1.0)
    
    # # gas/break also depend on lidar
    if deviation[8:12].max() < STUCK_THRESH:
        # go backwards if needed
        gas = 0
        bbreak = 1
        steer = 0
        return np.array([gas, bbreak, steer])
    else:
        # gas, but less intense as border approaches
        forward_dist = deviation[8:12].min()
        gas = min(1, forward_dist / 140)
        bbreak = 0

    # turning condition
    if deviation[8:12].min() < 170:
        steer_left_sum = deviation[:3].sum()
        steer_right_sum = deviation[-3:].sum()
        # select gas based on distance, gas from 0.3 to 1
        gas = 0.3 + (1 - 0.3) * deviation[8:12].min() / 170
        bbreak = 0
        # determine where to steer
        if steer_left_sum < steer_right_sum:
            steer = -0.7
        else:
            steer = 0.7

    return np.array([gas, bbreak, steer])


# Let us retrieve the TMRL Gymnasium environment.
# The environment you get from get_environment() depends on the content of config.json
env = get_environment()

sleep(1.0)  # just so we have time to focus the TM20 window after starting the script

print(env.action_space)
obs, info = env.reset()  # reset environment
for _ in range(2000):  # rtgym ensures this runs at 20Hz by default
    act = model(obs)  # compute action
    print(act)
    obs, rew, terminated, truncated, info = env.step(
        act
    )  # step (rtgym ensures healthy time-steps)
    print(f"Reward: {rew}")
    print(f"Finish: {terminated}")
    print(f"{info =}")
    if terminated or truncated:
        break
env.wait()  # rtgym-specific method to artificially 'pause' the environment when needed
