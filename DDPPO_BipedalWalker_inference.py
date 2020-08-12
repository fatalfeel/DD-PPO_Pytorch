from DDPPO_BipedalWalker_train import GameContent, CPPO_00, CPPO_01
from PIL import Image
import torch
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    ############## Hyperparameters ##############
    env_name        = "BipedalWalker-v3"
    env             = gym.make(env_name)
    dim_states      = env.observation_space.shape[0]
    dim_acts        = env.action_space.shape[0]
    
    n_episodes      = 10000         # num of episodes to run
    max_timesteps   = 1500          # max timesteps in one episode
    render          = True          # render the environment
    save_gif        = False         # png images are saved in gif folder
    train_epochs    = 40            # update policy for K epochs
    action_std      = 0.5           # constant std for action distribution (Multivariate Normal)
    lr              = 0.0001  # parameters for Adam optimizer
    betas           = (0.9, 0.999)
    gamma           = 0.99  # discount factor
    eps_clip        = 0.2           # clip parameter for CPPO
    vloss_coef      = 0.5  # clip parameter for PPO2
    entropy_coef    = 0.01
    #############################################
    
    gamedata    = GameContent()
    ppo         = CPPO_00(dim_states, dim_acts, action_std, lr, betas, gamma, train_epochs, eps_clip, vloss_coef, entropy_coef)
    #ppo        = CPPO_01(dim_states, dim_acts, action_std, lr, betas, gamma, train_epochs, eps_clip, vloss_coef, entropy_coef)

    directory = "./preTrained/"
    filename = "DDPPO_{}.pth".format(env_name)
    # map_location=torch.device('cpu') for cpu only if you have cuda then cancel it
    ppo.policy_ac.load_state_dict(torch.load(directory+filename, map_location=torch.device('cpu')))

    for ep in range(1, n_episodes+1):
        ep_reward = 0
        envstates = env.reset()
        for t in range(max_timesteps):
            #action = ppo.select_action(envstates, gamedata)
            action = ppo.policy_ac.interact(envstates, gamedata)
            envstates, reward, done, _ = env.step(action)
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
