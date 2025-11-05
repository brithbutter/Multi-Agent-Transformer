import torch

from mat.algorithms.actor_critic import Actor, Critic
from .ppo_policy import PPO_Policy
class MOPPO_Policy(PPO_Policy):
    def __init__(self, args, obs_space, cent_obs_space, act_space, n_objective = 2, device=torch.device("cpu")):
        self.args=args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        if self.args.use_cent_local_observe:
            self.obs_space = [obs_space[0] + cent_obs_space[0]]
        else:
            self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = Critic(args, self.share_obs_space, value_shape=n_objective, device=self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
    def save(self, save_dir, episode):
        torch.save(self.actor.state_dict(), str(save_dir) + "/moppo_" + str(episode) + ".pt")
    