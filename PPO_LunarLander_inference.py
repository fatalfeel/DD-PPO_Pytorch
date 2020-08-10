import gym
from PPO_LunarLander_train import GameContent, CPPO
from PIL import Image
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    ############## Hyperparameters ##############
    env_name        = "LunarLander-v2"
    # creating environment
    render          = True
    save_gif        = False
    n_episodes      = 10000
    h_neurons       = 64                # number of variables in hidden layer
    max_timesteps   = 400               # move 200 times rest game
    train_epochs    = 4                 # update policy for K epochs
    lr              = 0.0005
    gamma           = 0.99              # discount factor
    eps_clip        = 0.2               # clip parameter for PPO
    betas           = (0.9, 0.999)
    #############################################

    # creating environment
    env = gym.make(env_name)
    dim_states = env.observation_space.shape[0]
    dim_acts = 4

    gamedata    = GameContent()
    ppo         = CPPO(dim_states, dim_acts, h_neurons, lr, gamma, train_epochs, eps_clip, betas)

    directory   = "./preTrained/"
    filename    = "PPO_{}.pth".format(env_name)
    # map_location=torch.device('cpu') for cpu only if you have cuda then cancel it
    ppo.policy_ac.load_state_dict(torch.load(directory+filename, map_location=torch.device('cpu')))
    
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        estates = env.reset()
        for t in range(max_timesteps):
            action                      = ppo.policy_ac.interact(estates, gamedata)
            estates, reward, done, _    = env.step(action)
            ep_reward += reward
            if render:
                env.render()
            if save_gif:
                 img = env.render(mode = 'rgb_array')
                 img = Image.fromarray(img)
                 img.save('./gif/{}.jpg'.format(t))  
            if done:
                break
            
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        env.close()

    
