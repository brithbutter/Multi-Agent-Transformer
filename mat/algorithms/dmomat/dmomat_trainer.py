import numpy as np
import torch
import torch.nn as nn
from mat.utils.util import get_gard_norm, huber_loss, mse_loss
from mat.utils.valuenorm import ValueNorm
from mat.algorithms.utils.util import check
def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()
def normalize_advantage(adv,adv_copy):
    mean_advantages = np.nanmean(adv_copy)
    std_advantages = np.nanstd(adv_copy)
    return (adv - mean_advantages) / (std_advantages + 1e-5)
class DMOMATTrainer:
    """
    Trainer class for MAT to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 num_agents,
                 device=torch.device("cpu"),
                 n_objective = 2):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.num_agents = num_agents
        self.n_rollout_threads = args.n_rollout_threads
        self.n_objective = n_objective

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_actor_masks = args.use_actor_masks
        self.dec_actor = args.dec_actor
        self.single_dim_advantage = args.single_dim_advantage
        
        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(self.n_objective, device=self.device)
        else:
            self.value_normalizer = None
    def cal_single_value_loss(self, error_clipped, error_original, active_masks_batch):
        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)
        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()
        return value_loss
    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)

        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self.n_objective == 1:
            value_loss = self.cal_single_value_loss(error_clipped, error_original, active_masks_batch)
            return value_loss
        else:
            value_losses = []
            for i_objective in range(self.n_objective):
                value_loss = self.cal_single_value_loss(error_clipped[:,i_objective], error_original[:,i_objective], active_masks_batch[:,i_objective])
                value_losses.append(value_loss)
            single_error_clipped = (error_clipped**2).sum(dim=-1, keepdim=True).sqrt()
            single_error_original = (error_original**2).sum(dim=-1, keepdim=True).sqrt()
            single_value_loss = self.cal_single_value_loss(single_error_clipped, single_error_original, active_masks_batch[:,:1])
            return value_losses,single_value_loss

    def ppo_update(self, sample):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batchs, return_batchs, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targs, available_actions_batch = sample
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        adv_targs = check(adv_targs).to(**self.tpdv)
        value_preds_batchs = check(value_preds_batchs).to(**self.tpdv)
        return_batchs = check(return_batchs).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                            obs_batch, 
                                                                            rnn_states_batch, 
                                                                            rnn_states_critic_batch, 
                                                                            actions_batch, 
                                                                            masks_batch, 
                                                                            available_actions_batch,
                                                                            active_masks_batch,
                                                                            n_objective = self.n_objective)
        
        tot_policy_loss = 0
        policy_losses = []
        tot_value_loss = 0
        value_losses = []
        # imp_weights = torch.exp(torch.mean(action_log_probs,dim=-1,keepdim=True) - torch.mean(old_action_log_probs_batch,dim=-1,keepdim=True)).squeeze(-1)
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        # actor update
        

        available_actions_batch = check(available_actions_batch[:,-1]).to(**self.tpdv)
        # adv_targ = adv_targs.squeeze(-1)
        adv_targ = adv_targs
        value_preds_batch = value_preds_batchs
        return_batch = return_batchs
        value = values
        
        p1_coefficients = [1,1,1]
        p2_coefficients = [1,1,1]
        p_c_i = 0
        
        if self.n_objective == 1:
            value_loss = self.cal_value_loss(value, value_preds_batch, return_batch, active_masks_batch)
            value_losses.append(value_loss)
        else:
            value_losses,single_value_loss = self.cal_value_loss(value, value_preds_batch, return_batch, active_masks_batch)
            value_loss = None
            for loss,coef in zip(value_losses,self.value_loss_coef):
                if value_loss is None:
                    value_loss = loss * coef
                else:
                    value_loss += loss * coef
            value_loss = value_loss / self.n_objective
            # value_loss = self.value_loss_coef[0]*single_value_loss
        if self.single_dim_advantage:
            surr1 = imp_weights * adv_targ
            surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
            policy_loss = (-torch.min(surr1, surr2)* active_masks_batch[:,:1]).sum() / active_masks_batch[:,:1].sum() 
            policy_losses.append(policy_loss)
        else:
            for i_objective in range(self.n_objective):
            # for i_objective in range(self.n_objective):
                surr1 = imp_weights * adv_targ[:,i_objective].unsqueeze(-1)
                surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ[:,i_objective].unsqueeze(-1)
                if self._use_actor_masks:
                    # policy_loss = (-torch.sum(torch.min(surr1[:,i_objective], surr2[:,i_objective])* available_actions_batch[:],
                    #                         dim=-1,
                    #                         keepdim=True) ).sum() / available_actions_batch[:].sum() 
                    policy_loss = (-torch.min(surr1, surr2)* available_actions_batch[:]).sum() / available_actions_batch[:].sum() 
                elif self._use_policy_active_masks:
                    policy_loss = (-torch.sum(torch.min(surr1, surr2)* active_masks_batch,
                                            dim=-1,
                                            keepdim=True) ).sum() / active_masks_batch.sum()
                else:
                    policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
                policy_losses.append(policy_loss)
            
            
                if tot_policy_loss == 0:
                    tot_policy_loss = policy_loss * p1_coefficients[p_c_i]
                    # tot_value_loss = value_loss[:,i_objective]
                else:
                    tot_policy_loss += policy_loss * p2_coefficients[p_c_i]
                    # tot_value_loss += value_loss[:,i_objective]
            p_c_i += 1
            p_c_i = p_c_i % len(p1_coefficients)
        if self.single_dim_advantage:
            loss = policy_loss - dist_entropy * self.entropy_coef + value_loss 
        else:
            loss = (tot_policy_loss/self.n_objective) - dist_entropy * self.entropy_coef + value_loss 
        # loss = (tot_policy_loss/self.n_objective) - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef
        # loss = value_loss * self.value_loss_coef

        self.policy.optimizer.zero_grad()
        loss.backward()

        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.transformer.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_gard_norm(self.policy.transformer.parameters())

        self.policy.optimizer.step()

        return value_losses, grad_norm, policy_losses, dist_entropy, grad_norm, imp_weights

    def train(self, buffer):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        
        

        train_info = {}
        for i_objective in range(self.n_objective):
            train_info['value_loss_{}'.format(i_objective)] = 0
            if self.single_dim_advantage:
                train_info['policy_loss'] = 0
            else:
                train_info['policy_loss_{}'.format(i_objective)] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            
            # if buffer.available_actions is None:
            #     next_values = self.policy.get_values(np.concatenate(buffer.share_obs[-1]),
            #                                                 np.concatenate(buffer.obs[-1]),
            #                                                 np.concatenate(buffer.rnn_states_critic[-1]),
            #                                                 np.concatenate(buffer.masks[-1]))
            # else:
            #     next_values = self.policy.get_values(np.concatenate(buffer.share_obs[-1]),
            #                                                 np.concatenate(buffer.obs[-1]),
            #                                                 np.concatenate(buffer.rnn_states_critic[-1]),
            #                                                 np.concatenate(buffer.masks[-1]),
            #                                                 np.concatenate(buffer.available_actions[-1]),
            #                                                 n_objective = self.n_objective)
            # next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
            # buffer.compute_returns(next_values, self.value_normalizer)
            advantages_copy = buffer.advantages.copy()
            objective_coefficients_copy = buffer.objective_coefficients.copy()
            if buffer.single_dim_advantage:
                advantages_copy[buffer.active_masks[:-1,:,:,:1] == 0.0] = np.nan
            else:
                advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
            advantages = np.zeros_like(buffer.advantages)
            if buffer.single_dim_advantage:
                advantages = normalize_advantage(buffer.advantages, advantages_copy)
            else:
                for i in range(self.n_objective):
                    # mean_advantages = np.nanmean(advantages_copy[:,:,:,i])
                    # std_advantages = np.nanstd(advantages_copy[:,:,:,i])
                    # advantages[:,:,:,i] = (buffer.advantages[:,:,:,i] - mean_advantages) / (std_advantages + 1e-5)
                    advantages[:,:,:,i] = normalize_advantage(buffer.advantages[:,:,:,i], advantages_copy[:,:,:,i])
            # for row in range(advantages.shape[0]):
            #     for col in range(advantages.shape[1]):
            #         advantages[row,col] = advantages[row,col] * objective_coefficients_copy[row,col]
            
            data_generator = buffer.feed_forward_generator_transformer(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_losses, critic_grad_norm, policy_losses, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample)

                for i_objective in range(self.n_objective):
                    train_info['value_loss_{}'.format(i_objective)] += value_losses[i_objective].item()
                    if self.single_dim_advantage:
                        train_info['policy_loss'] += policy_losses[0].item()
                    else:
                        train_info['policy_loss_{}'.format(i_objective)] += policy_losses[i_objective].item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.train()

    def prep_rollout(self):
        self.policy.eval()
