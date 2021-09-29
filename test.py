# Single-stage manipulation environments
# Reach, Push, PickAndPlace, Slide
import pybullet_multigoal_gym as pmg
import numpy as np
# Install matplotlib if you want to use imshow to view the goal images
import matplotlib.pyplot as plt

camera_setup = [
    {
        'cameraEyePosition': [-1.0, 0.25, 0.6],
        'cameraTargetPosition': [-0.6, 0.05, 0.2],
        'cameraUpVector': [0, 0, 1],
        'render_width': 128,
        'render_height': 128
    },
    {
        'cameraEyePosition': [-1.0, -0.25, 0.6],
        'cameraTargetPosition': [-0.6, -0.05, 0.2],
        'cameraUpVector': [0, 0, 1],
        'render_width': 128,
        'render_height': 128
    }
]

env = pmg.make_env(
    # task args ['reach', 'push', 'slide', 'pick_and_place',
    #            'block_stack', 'block_rearrange', 'chest_pick_and_place', 'chest_push']
    task='chest_pick_and_place',
    gripper='robotiq85',
    num_block=4,  # only meaningful for multi-block tasks
    render=True,
    binary_reward=True,
    distance_threshold=0.1,
    max_episode_steps=100000,
    # image observation args
    image_observation=False,
    depth_image=False,
    goal_image=True,
    visualize_target=True,
    camera_setup=camera_setup,
    observation_cam_id=0,
    goal_cam_id=1, )
    # curriculum args
    # use_curriculum=False,
    # num_goals_to_generate=90)

obs = env.reset()
action = env.action_space.sample()
# action = np.array([0, 0, 0.6, 1])
t = 0
while True:
    t += 1
    # print(t)
    # action = env.action_space.sample()
    a = t / 100
    r = 0.05
    action = np.array([-0.5 + np.cos(a) * r, 0 + np.sin(a) * r, 0.2, 0, 0, 0, np.cos(a)])
    obs, reward, done, info = env.step(action)
    # print('state: ', obs['state'], '\n',
    #       'desired_goal: ', obs['desired_goal'], '\n',
    #       'achieved_goal: ', obs['achieved_goal'], '\n',
    #       'reward: ', reward, '\n')
    # print(obs['desired_goal'])
    # plt.imshow(obs['observation'])
    # plt.pause(0.00001)
    # print(reward)
    if done:
        env.reset()