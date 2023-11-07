import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        # print("obs_space: ", self.envs.observation_space)
        # print("share_obs_space: ", self.envs.share_observation_space)
        # print("act_space: ", self.envs.action_space)

        if self.all_args.algorithm_name == "happo" or self.all_args.algorithm_name == "hatrpo":
            from mat.utils.separated_buffer import SeparatedReplayBuffer
            if self.all_args.algorithm_name == "happo":
                from mat.algorithms.mat.happo_trainer import HAPPO as TrainAlgo
                from mat.algorithms.mat.algorithm.happo_policy import HAPPO_Policy as Policy
            else:
                from mat.algorithms.hatrpo.hatrpo_trainer import HATRPO as TrainAlgo
                from mat.algorithms.hatrpo.hatrpo_policy import HATRPO_Policy as Policy
            self.policy = []
            for agent_id in range(self.num_agents):
                if len(self.envs.observation_space)==1:
                    index = 0
                else:
                    index = agent_id
                share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[index]
                # policy network
                po = Policy(self.all_args,
                            self.envs.observation_space[index],
                            share_observation_space,
                            self.envs.action_space[index],
                            device = self.device)
                self.policy.append(po)


            self.trainer = []
            self.buffer = []
            for agent_id in range(self.num_agents):
                # algorithm
                tr = TrainAlgo(self.all_args, self.policy[agent_id], device = self.device)
                # buffer
                if len(self.envs.observation_space)==1:
                    index = 0
                else:
                    index = agent_id
                share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[index]
                bu = SeparatedReplayBuffer(self.all_args,
                                        self.envs.observation_space[index],
                                        share_observation_space,
                                        self.envs.action_space[index])
                self.buffer.append(bu)
                self.trainer.append(tr)
        elif self.algorithm_name == "rmappo":
            from mat.utils.separated_buffer import SeparatedReplayBuffer
            from mat.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from mat.algorithms.r_mappo.rMAPPOPolicy import R_MAPPOPolicy as Policy
            self.policy = []
            for agent_id in range(self.num_agents):
                if len(self.envs.observation_space)==1:
                    index = 0
                else:
                    index = agent_id
                share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[index]
                # policy network
                po = Policy(self.all_args,
                            self.envs.observation_space[index],
                            share_observation_space,
                            self.envs.action_space[index],
                            device = self.device)
                self.policy.append(po)


            self.trainer = []
            self.buffer = []
            for agent_id in range(self.num_agents):
                # algorithm
                tr = TrainAlgo(self.all_args, self.policy[agent_id], device = self.device)
                # buffer
                if len(self.envs.observation_space)==1:
                    index = 0
                else:
                    index = agent_id
                share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[index]
                bu = SeparatedReplayBuffer(self.all_args,
                                        self.envs.observation_space[index],
                                        share_observation_space,
                                        self.envs.action_space[index])
                self.buffer.append(bu)
                self.trainer.append(tr)
        elif self.all_args.algorithm_name == "random":
            from mat.algorithms.random.algorithm.random_policy import Random_Policy as Policy
            from mat.algorithms.random.random_trainer import RandomTrainer as TrainAlgo
            from mat.utils.shared_buffer import SharedReplayBuffer
            self.policy = Policy(self.all_args,
                                self.envs.observation_space[0],
                                share_observation_space,
                                self.envs.action_space[0],
                                self.num_agents,
                                device=self.device)
            self.trainer = TrainAlgo(self.all_args, self.policy, self.num_agents, device=self.device)
            self.buffer = SharedReplayBuffer(self.all_args,
                                            self.num_agents,
                                            self.envs.observation_space[0],
                                            share_observation_space,
                                            self.envs.action_space[0],
                                            self.all_args.env_name)
        else:
            from mat.utils.shared_buffer import SharedReplayBuffer
            from mat.algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
            from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy
            # policy network
            self.policy = Policy(self.all_args,
                                self.envs.observation_space[0],
                                share_observation_space,
                                self.envs.action_space[0],
                                self.num_agents,
                                device=self.device)

        
            # algorithm
            self.trainer = TrainAlgo(self.all_args, self.policy, self.num_agents, device=self.device)
            
            # buffer
            self.buffer = SharedReplayBuffer(self.all_args,
                                            self.num_agents,
                                            self.envs.observation_space[0],
                                            share_observation_space,
                                            self.envs.action_space[0],
                                            self.all_args.env_name)
        if self.model_dir is not None:
            self.restore(self.model_dir)

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        if self.all_args.algorithm_name == "happo" or self.all_args.algorithm_name == "rmappo" or self.all_args.algorithm_name == "hatrpo":
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1], 
                                                                    self.buffer[agent_id].rnn_states_critic[-1],
                                                                    self.buffer[agent_id].masks[-1])
                next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)
        elif self.all_args.algorithm_name == "random":
            pass
        else:
            self.trainer.prep_rollout()
            if self.buffer.available_actions is None:
                next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                            np.concatenate(self.buffer.obs[-1]),
                                                            np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                            np.concatenate(self.buffer.masks[-1]))
            else:
                next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                            np.concatenate(self.buffer.obs[-1]),
                                                            np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                            np.concatenate(self.buffer.masks[-1]),
                                                            np.concatenate(self.buffer.available_actions[-1]))
            next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
            self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        if self.all_args.algorithm_name == "happo" or self.all_args.algorithm_name == "rmappo" or self.all_args.algorithm_name == "hatrpo":
            train_infos = []
            # random update order

            action_dim=self.buffer[0].actions.shape[-1]
            factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

            for agent_id in torch.randperm(self.num_agents):
                self.trainer[agent_id].prep_training()
                self.buffer[agent_id].update_factor(factor)
                available_actions = None if self.buffer[agent_id].available_actions is None \
                    else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])
                
                if self.all_args.algorithm_name == "hatrpo":
                    old_actions_logprob, _, _, _, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                        self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                        self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                        self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                        available_actions,
                                                        self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
                else:
                    if self.all_args.use_cent_local_observe:
                        old_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(
                            np.concatenate((self.buffer[agent_id].share_obs[:-1].reshape(-1, *self.buffer[agent_id].share_obs.shape[2:]),
                                            self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:])),axis=-1),
                                                        self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                        self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                        self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                        available_actions,
                                                        self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
                    else:
                        old_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
                train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            
                if self.all_args.algorithm_name == "hatrpo":
                    new_actions_logprob, _, _, _, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                        self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                        self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                        self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                        available_actions,
                                                        self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
                else:
                    if self.all_args.use_cent_local_observe:
                        new_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(
                                                np.concatenate((self.buffer[agent_id].share_obs[:-1].reshape(-1, *self.buffer[agent_id].share_obs.shape[2:]),
                                                                self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:])),axis=-1),
                                                self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                available_actions,
                                                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
                    else:
                        new_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                    self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                    self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                    self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                    available_actions,
                                                    self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))

                factor = factor*_t2n(torch.prod(torch.exp(new_actions_logprob-old_actions_logprob),dim=-1).reshape(self.episode_length,self.n_rollout_threads,1))
                train_infos.append(train_info)      
                self.buffer[agent_id].after_update()
            
            return train_infos
        if self.all_args.algorithm_name == "random":
            train_infos = self.trainer.train(self.buffer)      
            return train_infos
        else:
            self.trainer.prep_training()
            train_infos = self.trainer.train(self.buffer)      
            self.buffer.after_update()
            return train_infos

    def save(self, episode):
        """Save policy's actor and critic networks."""
        if self.all_args.algorithm_name == "happo" or self.algorithm_name == "rmappo" or self.all_args.algorithm_name == "hatrpo":
            for agent_id in range(self.num_agents):
                if self.use_single_network:
                    policy_model = self.trainer[agent_id].policy.model
                    torch.save(policy_model.state_dict(), str(self.save_dir) + "/model_agent" + str(agent_id) + ".pt")
                else:
                    policy_actor = self.trainer[agent_id].policy.actor
                    torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
                    policy_critic = self.trainer[agent_id].policy.critic
                    torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")
        elif self.all_args.algorithm_name == "random":
            pass
        else:
            self.policy.save(self.save_dir, episode)

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        if self.all_args.algorithm_name == "happo" or self.algorithm_name == "rmappo" or self.all_args.algorithm_name == "hatrpo":
            for agent_id in range(self.num_agents):
                if self.use_single_network:
                    policy_model_state_dict = torch.load(str(self.model_dir) + '/model_agent' + str(agent_id) + '.pt')
                    self.policy[agent_id].model.load_state_dict(policy_model_state_dict)
                else:
                    policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
                    self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
                    policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
                    self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)
        else:
            self.policy.restore(model_dir)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        if self.all_args.algorithm_name == "happo" or self.algorithm_name == "rmappo" or self.all_args.algorithm_name == "hatrpo":
            # for agent_id in range(self.num_agents):
            #     for k, v in train_infos[agent_id].items():
            #         agent_k = "agent%i/" % agent_id + k
            #         self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
        
            for k, v in train_infos[0].items():
                agent_k = "agent%i/" % 0 + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
        else:
            for k, v in train_infos.items():
                if self.use_wandb:
                    wandb.log({k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
