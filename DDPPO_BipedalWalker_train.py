import numpy
import torch
import torch.nn as nn
import torch.distributions
import gym
from multiprocessing import Process

device = torch.device("cpu")

class GameContent:
    def __init__(self):
        self.actions        = []
        self.states         = []
        self.actorlogprobs  = []
        self.rewards        = []
        self.is_terminals   = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.actorlogprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class Actor_Critic(nn.Module):
    def __init__(self, dim_states, dim_acts, action_std):
        super(Actor_Critic, self).__init__()
        # action mean range -1 to 1
        self.network_act =  nn.Sequential(  nn.Linear(dim_states, 64),
                                            nn.Tanh(),
                                            nn.Linear(64, 32),
                                            nn.Tanh(),
                                            nn.Linear(32, dim_acts),
                                            nn.Tanh()  )
        # network_value
        self.network_critic = nn.Sequential(nn.Linear(dim_states, 64),
                                            nn.Tanh(),
                                            nn.Linear(64, 32),
                                            nn.Tanh(),
                                            nn.Linear(32, 1) )

        #self.action_std = torch.full((dim_acts,), action_std*action_std).double().to(device)
        self.action_std  = torch.full((dim_acts,), action_std).double().to(device) #standard deviations

    def forward(self):
        raise NotImplementedError

    # https://pytorch.org/docs/stable/distributions.html
    # backpropagation conditions are continue and differential. Sampling probs need in one distribution
    def interact(self, envstate, gamedata):
        torchstate      = torch.DoubleTensor(envstate.reshape(1, -1)).to(device) #reshape(1,-1) 1d to 2d
        act_mu          = self.network_act(torchstate)
        std_mat         = torch.diag(self.action_std).double().to(device) #transfer to matrix
        #distribute     = torch.distributions.MultivariateNormal(action_mu, cov_mat)
        distribute      = torch.distributions.MultivariateNormal(act_mu, scale_tril=std_mat) #act_mu=center, scale_tril=width
        action          = distribute.sample()
        actlogprob      = distribute.log_prob(action) #logeX

        gamedata.states.append(torchstate)
        gamedata.actions.append(action)
        gamedata.actorlogprobs.append(actlogprob)

        #flatten do 2d to 1d
        return action.detach().cpu().data.numpy().flatten()

    # policy_ac.calculation will call
    # sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
    # usually mini_batch_size sample < states'size do forward
    # in our example the mini_batch_size = states'size
    def calculation(self, states, actions):
        acts_mu             = self.network_act(states)
        acts_std            = self.action_std.expand_as(acts_mu)
        mats_std            = torch.diag_embed(acts_std).double().to(device)
        #distribute         = torch.distributions.MultivariateNormal(action_mu, cov_mat)
        distribute          = torch.distributions.MultivariateNormal(acts_mu, scale_tril=mats_std) #act_mu=center, scale_tril=width
        critic_actlogprobs  = distribute.log_prob(actions) #logeX
        entropy             = distribute.entropy() #entropy is uncertain percentage, value higher mean uncertain more
        next_critic_values  = self.network_critic(states) #c_values is V(s) in A3C theroy

        #if dimension can squeeze then tensor 3d to 2d.
        #EX: squeeze tensor[2,1,3] become to tensor[2,3]
        return critic_actlogprobs, torch.squeeze(next_critic_values), entropy

    # if is_terminals is false use Markov formula to replace last reward
    def predict_reward(self, next_state, gamedata, gamma):
        if gamedata.is_terminals[-1] is False:
            torchstate = torch.DoubleTensor(next_state.reshape(1, -1)).to(device)  # reshape(1,-1) 1d to 2d
            next_value = self.network_critic(torchstate)
            data_value = next_value.detach().cpu().data.numpy()[0,0]
            gamedata.rewards[-1] = gamedata.rewards[-1] + gamma * data_value

class CPPO_00:
    def __init__(self, dim_states, dim_acts, action_std, lr, gamma, train_epochs, eps_clip, betas):
        self.lr             = lr
        self.betas          = betas
        self.gamma          = gamma
        self.eps_clip       = eps_clip
        self.train_epochs   = train_epochs

        self.policy_ac      = Actor_Critic(dim_states, dim_acts, action_std).double().to(device)
        self.optimizer      = torch.optim.Adam(self.policy_ac.parameters(), lr=lr, betas=betas)

        #self.policy_curr = Actor_Critic(dim_states, dim_acts, action_std).double().to(device)
        #self.policy_curr.load_state_dict(self.policy_ac.state_dict())

        self.MseLoss        = nn.MSELoss(reduction='none')

    #def select_action(self, estates, gamedata):
    #    tstates = torch.DoubleTensor(estates.reshape(1, -1)).double().to(device)
    #    return self.policy_curr.interact(tstates, gamedata).cpu().data.numpy().flatten()

    def Average_Gradient(self, model):
        size = numpy.float64(torch.distributed.get_world_size())
        for param in model.parameters():
            # print(rank, param.grad.data, "\n")
            torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            # print(rank, param.grad.data, "\n")
            param.grad.data /= size
            # print(rank, param.grad.data, "\n")

    def train_update(self, gamedata):
        returns             = []
        discounted_reward   = 0
        # Monte Carlo estimate of returns:
        for reward, is_terminal in zip(reversed(gamedata.rewards), reversed(gamedata.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns).double().to(device)

        # convert list to tensor
        curr_states         = torch.squeeze(torch.stack(gamedata.states).double().to(device), 1).detach()
        curr_actions        = torch.squeeze(torch.stack(gamedata.actions).double().to(device), 1).detach()
        curr_actlogprobs    = torch.squeeze(torch.stack(gamedata.actorlogprobs).double().to(device), 1).detach()

        # critic_state_reward   = network_critic(curraccu_states)
        '''refer to a2c-ppo should modify like this
           advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
           advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)'''
        critic_vpi  = self.policy_ac.network_critic(curr_states)
        critic_vpi  = torch.squeeze(critic_vpi)
        qsa_sub_vs  = returns - critic_vpi.detach()  # A(s,a) => Q(s,a) - V(s), V(s) is critic
        advantages  = (qsa_sub_vs - qsa_sub_vs.mean()) / (qsa_sub_vs.std() + 1e-5)

        # Optimize policy for K epochs:
        for _ in range(self.train_epochs):
            #cstate_value is V(s) in A3C theroy. critic network is another actor input state
            critic_actlogprobs, next_critic_values, entropy = self.policy_ac.calculation(curr_states, curr_actions)

            # https://socratic.org/questions/what-is-the-derivative-of-e-lnx
            # log(critic) - log(curraccu) = log(critic/curraccu)
            # ratios  = e^(ln(State2_actProbs)-ln(State1_actProbs)) =  e^ln(State2_actProbs/State1_actProbs)
            # ratios  = (State2_critic_actProbs/State1_actor_actProbs)
            # ratios  = next_critic_actprobs/curr_actions_prob = Pw(A1|S2)/Pw(A1|S1), where w is weights(theta)
            ratios  = torch.exp(critic_actlogprobs - curr_actlogprobs.detach())

            #advantages is stdscore mode
            surr1   = ratios * advantages
            surr2   = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # mseLoss is Mean Square Error = (target - output)^2, next_critic_values in first param follow libtorch rules
            loss    = -torch.min(surr1, surr2) + 0.5*self.MseLoss(next_critic_values, returns) - 0.01*entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()

            self.Average_Gradient(self.policy_ac)

            self.optimizer.step()

        # Copy new weights into old policy:
        #self.policy_curr.load_state_dict(self.policy_ac.state_dict())

class CPPO_01:
    def __init__(self, dim_states, dim_acts, action_std, lr, gamma, train_epochs, eps_clip, betas):
        self.lr             = lr
        self.betas          = betas
        self.gamma          = gamma
        self.eps_clip       = eps_clip
        self.train_epochs   = train_epochs

        self.policy_ac      = Actor_Critic(dim_states, dim_acts, action_std).double().to(device)
        self.optimizer      = torch.optim.Adam(self.policy_ac.parameters(), lr=lr, betas=betas)

        #self.policy_curr = Actor_Critic(dim_states, dim_acts, action_std).double().to(device)
        #self.policy_curr.load_state_dict(self.policy_ac.state_dict())

        self.MseLoss        = nn.MSELoss(reduction='none')

    #def select_action(self, estates, gamedata):
    #    tstates = torch.DoubleTensor(estates.reshape(1, -1)).double().to(device)
    #    return self.policy_curr.interact(tstates, gamedata).cpu().data.numpy().flatten()

    def Average_Gradient(self, model):
        size = numpy.float64(torch.distributed.get_world_size())
        for param in model.parameters():
            # print(rank, param.grad.data, "\n")
            torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            # print(rank, param.grad.data, "\n")
            param.grad.data /= size
            # print(rank, param.grad.data, "\n")

    def train_update(self, gamedata):
        returns             = []
        discounted_reward   = 0
        # Monte Carlo estimate of returns:
        for reward, is_terminal in zip(reversed(gamedata.rewards), reversed(gamedata.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns).double().to(device)

        # convert list to tensor
        curr_states         = torch.squeeze(torch.stack(gamedata.states).double().to(device), 1).detach()
        curr_actions        = torch.squeeze(torch.stack(gamedata.actions).double().to(device), 1).detach()
        curr_actlogprobs    = torch.squeeze(torch.stack(gamedata.actorlogprobs).double().to(device), 1).detach()

        # critic_state_reward   = network_critic(curraccu_states)
        '''refer to a2c-ppo should modify like this
           advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
           advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)'''
        critic_vpi  = self.policy_ac.network_critic(curr_states)
        critic_vpi  = torch.squeeze(critic_vpi)
        qsa_sub_vs  = returns - critic_vpi.detach()  # A(s,a) => Q(s,a) - V(s), V(s) is critic
        advantages  = (qsa_sub_vs - qsa_sub_vs.mean()) / (qsa_sub_vs.std() + 1e-5)

        # Optimize policy for K epochs:
        for _ in range(self.train_epochs):
            #cstate_value is V(s) in A3C theroy. critic network is another actor input state
            critic_actlogprobs, next_critic_values, entropy = self.policy_ac.calculation(curr_states, curr_actions)

            # https://socratic.org/questions/what-is-the-derivative-of-e-lnx
            # log(critic) - log(curraccu) = log(critic/curraccu)
            # ratios  = e^(ln(State2_actProbs)-ln(State1_actProbs)) =  e^ln(State2_actProbs/State1_actProbs)
            # ratios  = (State2_critic_actProbs/State1_actor_actProbs)
            # ratios  = next_critic_actprobs/curr_actions_prob = Pw(A1|S2)/Pw(A1|S1), where w is weights(theta)
            ratios  = torch.exp(critic_actlogprobs - curr_actlogprobs.detach())

            #advantages is stdscore mode
            surr1   = ratios * advantages
            surr2   = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # mseLoss is Mean Square Error = (target - output)^2, next_critic_values in first param follow libtorch rules
            loss    = -torch.min(surr1, surr2) + 0.5*self.MseLoss(next_critic_values, returns) - 0.01*entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()

            self.Average_Gradient(self.policy_ac)

            self.optimizer.step()

        # Copy new weights into old policy:
        #self.policy_curr.load_state_dict(self.policy_ac.state_dict())

def BipedalWalker_00(rank):
    ############## Hyperparameters ##############
    env_name        = "BipedalWalker-v3"
    render          = False
    solved_reward   = 300           # stop training if avg_reward > solved_reward
    log_interval    = 20            # print avg reward in the interval
    max_episodes    = 500000        # max training episodes
    max_timesteps   = 1500          # max timesteps in one episode
    update_timestep = 4000          # train_update policy every n timesteps
    train_epochs    = 40            # train_update policy for K epochs
    action_std      = 0.5           # constant std for action distribution (Multivariate Normal)
    gamma           = 0.99          # discount factor
    lr              = 0.0001        # parameters for Adam optimizer
    eps_clip        = 0.2           # clip parameter for CPPO
    betas           = (0.9, 0.999)  # Adam β
    predict_trick   = True          # trick shot make PPO get better action & reward
    #############################################

    # creating environment
    env         = gym.make(env_name)
    dim_states  = env.observation_space.shape[0]
    dim_acts    = env.action_space.shape[0]

    gamedata    = GameContent()
    ppo         = CPPO_00(dim_states, dim_acts, action_std, lr, gamma, train_epochs, eps_clip, betas)

    # logging variables
    running_reward  = 0
    avg_length      = 0
    timestep        = 0

    # training loop
    for i_episode in range(1, max_episodes+1):
        envstate = env.reset()
        for t in range(max_timesteps):
            timestep += 1

            # Running policy_current: #Done-0 State-0 Act-0
            action = ppo.policy_ac.interact(envstate, gamedata)

            # Done-1 State-1 Act-0 R-0
            envstate, reward, done, _ = env.step(action)

            # Saving reward and is_terminals:
            gamedata.rewards.append(reward)

            # is_terminal in next state:
            gamedata.is_terminals.append(done)

            # train_update if its time
            if timestep % update_timestep == 0:
                if predict_trick is True:
                    ppo.policy_ac.predict_reward(envstate, gamedata, gamma)

                ppo.train_update(gamedata)
                gamedata.clear_memory()

                timestep = 0

            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## BipedalWalker_00 Solved! ##########")
            torch.save(ppo.policy_ac.state_dict(), './DDPPO_00_{}.pth'.format(env_name))
            break

        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy_ac.state_dict(), './DDPPO_00_{}_episode_{}.pth'.format(env_name, i_episode))

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

def BipedalWalker_01(rank):
    ############## Hyperparameters ##############
    env_name        = "BipedalWalker-v3"
    render          = False
    solved_reward   = 300           # stop training if avg_reward > solved_reward
    log_interval    = 20            # print avg reward in the interval
    max_episodes    = 500000        # max training episodes
    max_timesteps   = 1500          # max timesteps in one episode
    update_timestep = 4000          # train_update policy every n timesteps
    train_epochs    = 40            # train_update policy for K epochs
    action_std      = 0.5           # constant std for action distribution (Multivariate Normal)
    gamma           = 0.99          # discount factor
    lr              = 0.0001        # parameters for Adam optimizer
    eps_clip        = 0.2           # clip parameter for CPPO
    betas           = (0.9, 0.999)  # Adam β
    predict_trick   = True          # trick shot make PPO get better action & reward
    #############################################

    # creating environment
    env         = gym.make(env_name)
    dim_states  = env.observation_space.shape[0]
    dim_acts    = env.action_space.shape[0]

    gamedata    = GameContent()
    ppo         = CPPO_01(dim_states, dim_acts, action_std, lr, gamma, train_epochs, eps_clip, betas)

    # logging variables
    running_reward  = 0
    avg_length      = 0
    timestep        = 0

    # training loop
    for i_episode in range(1, max_episodes+1):
        envstate = env.reset()
        for t in range(max_timesteps):
            timestep += 1

            # Running policy_current: #Done-0 State-0 Act-0
            action = ppo.policy_ac.interact(envstate, gamedata)

            # Done-1 State-1 Act-0 R-0
            envstate, reward, done, _ = env.step(action)

            # Saving reward and is_terminals:
            gamedata.rewards.append(reward)

            # is_terminal in next state:
            gamedata.is_terminals.append(done)

            # train_update if its time
            if timestep % update_timestep == 0:
                if predict_trick is True:
                    ppo.policy_ac.predict_reward(envstate, gamedata, gamma)

                ppo.train_update(gamedata)
                gamedata.clear_memory()

                timestep = 0

            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy_ac.state_dict(), './DDPPO_01_{}.pth'.format(env_name))
            break

        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy_ac.state_dict(), './DDPPO_01_{}_episode_{}.pth'.format(env_name, i_episode))

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

#gloo for cpu
def init_processes_00(rank, world_size, fn, backend='gloo'):
    #os.environ['MASTER_ADDR'] = '127.0.0.1' #localhost ip
    #os.environ['MASTER_PORT'] = '6900'
    torch.distributed.init_process_group(backend,
                                         init_method='tcp://127.0.0.1:6901',
                                         rank=rank,
                                         world_size=world_size)
    fn(rank)
    torch.distributed.destroy_process_group()

#gloo for cpu
def init_processes_01(rank, world_size, fn, backend='gloo'):
    #os.environ['MASTER_ADDR'] = '127.0.0.1' #localhost ip
    #os.environ['MASTER_PORT'] = '6900'
    torch.distributed.init_process_group(backend,
                                         init_method='tcp://127.0.0.1:6901',
                                         rank=rank,
                                         world_size=world_size)
    fn(rank)
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    world_size  = 2  # num of processes
    threadings  = []
    init_func   = [None] * world_size
    exe_func    = [None] * world_size

    #split 2 functions is easier to debug than 2 threads in one function
    init_func[0] = init_processes_00
    init_func[1] = init_processes_01

    exe_func[0] = BipedalWalker_00
    exe_func[1] = BipedalWalker_01

    for rank in range(world_size):
        ps = Process(target=init_func[rank], args=(rank, world_size, exe_func[rank]))
        ps.start()
        threadings.append(ps)

    for ps in threadings:
        ps.join()  # wait to finish