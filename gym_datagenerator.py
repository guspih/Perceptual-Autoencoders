import numpy as np
import pickle
import random
import signal
import multiprocessing
import gym
import cv2
import os
import matplotlib.pyplot as plt

def init_worker():
    '''
    Setup worker to throw exceptions back to the main process
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def collect_rollout_data(environment, agent, timesteps, image_size):
    '''
    Runs one rollout in the given environment with the given agent
    Args:
        environment (str): ID of openai gym to run
        agent (f() -> Object / None): Agent policy. Random if None
        timesteps (int): Nr of timesteps to record rollout for
        image_size (int, int): Size of images to be stored in pixels
    Returns ([np.array],[float],[bool],[any],[any]): Data from each timestep
    '''
    imgs = []
    rewards = []
    dones = []
    actions = []
    observations = []
    rets = (imgs, rewards, dones, actions, observations)

    env = gym.make(environment)
    observation = env.reset()
    if not agent is None:
        actor = agent()
    for _ in range(timesteps):
        #Each timestep render the env, take an action and update env
        if environment != 'CarRacing-v0':
            img = env.render('rgb_array')
        else:
            img = observation
        if agent is None:
            action = env.action_space.sample()
        else:
            action = actor(observation)
        observation, reward, done, info = env.step(action)
        
        #Downsize, covert to float np.array, and store image
        small_image = np.array(
            np.true_divide(
                cv2.resize(
                    img, image_size, 
                    interpolation=cv2.INTER_CUBIC
                ),
                255
            ), 
            dtype = np.float16
        )

        #Collect data
        imgs.append(small_image)
        rewards.append(reward)
        dones.append(done)
        actions.append(action)
        if environment != 'CarRacing-v0':
            observations.append(observation)
    #Close environement and return data
    env.close()
    return rets

def generate_gym_data(
    environment='LunarLander-v2',
    rollouts=700,
    timesteps_per_rollout=150,
    image_size=(64,64),
    save_file=None,
    agent=None,
    workers=1
):
    '''
    Creates a .pickle file containing images, actions, parameters, etc
    of a number of rollouts in a given Gym environment
    Args:
        environment (str): ID of openai gym to run
        rollouts (int): How many runs will be recorded
        timesteps_per_rollout (int): Nr of timesteps recorded per rollout
        image_size (int, int): Size of images to be stored in pixels
        save_file (str / None): Name of the file to store the dataset in
        agent (f() -> Object / None): Agent policy. Random if None
    '''
    #Creating a save_file name if None is provided
    if save_file is None:
        save_file = f'{environment}_{rollouts*timesteps_per_rollout}.pickle'
        if not os.path.isdir('datasets/' + environment):
            os.mkdir('datasets/' + environment)
        save_file = 'datasets/' + environment + '/' + save_file

    #Init dict for data
    data = {
        'imgs' : [],
        'rewards' : [],
        'dones' : [],
        'actions' : [],
        'parameters' : {
            'environment' : environment,
            'rollouts' : rollouts,
            'timesteps_per_rollout' : timesteps_per_rollout,
            'image_size' : image_size,
            'agent' : agent.__class__.__name__
        }
    }
    if environment != 'CarRacing-v0':
        data['observations'] = []
    
    
    pool = multiprocessing.Pool(workers, init_worker)
    
    #Run several rollout in parallel
    try:
        processes = [
            pool.apply_async(
                collect_rollout_data, 
                (environment, agent, timesteps_per_rollout, image_size)
            )
            for _ in range(rollouts)
        ]
        for i, process in enumerate(processes):
            imgs, rewards, dones, actions, observations = process.get()
            data['imgs'] += imgs
            data['rewards'] += rewards
            data['dones'] += dones
            data['actions'] += actions
            if environment != 'CarRacing-v0':
                data['observations'] += observations
    except Exception as e:
        pool.close()
        pool.terminate()
        pool.join()
        raise e
    else:
        pool.close()
        pool.join()
    
    #Save all collected data and parameters in a .pickle file
    pickle.dump(data, open(save_file, 'wb'))

if __name__ == '__main__':
    '''
    If run directly this will generate data from the LunarLander-v2 environment
    '''
    generate_gym_data(
        rollouts=700,
        timesteps_per_rollout=150,
        workers=4
    )