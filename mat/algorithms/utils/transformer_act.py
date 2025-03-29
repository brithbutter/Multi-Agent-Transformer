import torch
from torch.distributions import Categorical, Normal
from torch.nn import functional as F
import time 

NORMAL_STD = 0.3

def discrete_action(logit,i=1,available_actions=None,last=False,deterministic=False):
    if last:
        action = F.one_hot(logit)[1]
        action_log = torch.zeros_like(action,dtype=torch.float32)
    else:
        if available_actions is not None:
            ava_a = available_actions[:, i, :]
            logit[available_actions[:, i, :] == 0] = -1e10
            distri = Categorical(logits=logit)
            action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
            action_log = distri.log_prob(action)
    return action,action_log

def continuous_action(act_mean,action_std,deterministic=False):
    distri = Normal(act_mean, action_std)
    action = act_mean if deterministic else distri.sample()
    action_log = distri.log_prob(action)
    return action,action_log

def semi_discrete_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                available_actions=None, deterministic=False, semi_index = -1,stride = 2):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim+1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.float32)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)
    
    if deterministic:
        starting_index = 0
        ending_index = 1
        continue_flag = True
        while ending_index <= n_agent and continue_flag:
            if ending_index == n_agent:
                continue_flag = False
            logits = decoder(shifted_action, obs_rep, obs)[:, starting_index:ending_index, :]
            for i in range(ending_index - starting_index):
                if i + starting_index < n_agent+semi_index:
                    logit = logits[:,i,:]
                    action,action_log = discrete_action(logit=logit,i=i+starting_index,available_actions=available_actions,last=(i+starting_index == n_agent-1),deterministic=deterministic)
                    output_action[:, i+starting_index, :] = action.unsqueeze(-1)
                    output_action_log[:, i+starting_index, :] = action_log.unsqueeze(-1)
                    if i + 1 < n_agent:
                        shifted_action[:, i + starting_index + 1, 1:] = F.one_hot(action, num_classes=action_dim)
                else:
                    act_mean = logits[:,i,:] 
                    action_std = torch.sigmoid(decoder.log_std) * 0.5
                    action,action_log = continuous_action(act_mean=act_mean,action_std=action_std,deterministic=deterministic)
                    output_action[:, i+starting_index, :] = action[:,1].reshape(-1,1)
                    output_action_log[:, i+starting_index, :] = action_log[:,1].reshape(-1,1)
                    if i +starting_index+ 1 < n_agent:
                        shifted_action[:, i +starting_index + 1, :] = action 
            if ending_index < n_agent+semi_index:
                starting_index = ending_index
                # ending_index *= 3
                ending_index += stride
                if ending_index > n_agent+semi_index:
                    ending_index = n_agent+semi_index
            else:
                starting_index = ending_index
                ending_index += 1
                if ending_index > n_agent:
                    ending_index = n_agent
    else:
        for i in range(n_agent):
            
            if i < n_agent+semi_index:
                # start_time = time.time()
                logit = decoder(shifted_action, obs_rep, obs)[:, i, :]
                # decod_time = time.time() - start_time
                action,action_log = discrete_action(logit=logit,i=i,available_actions=available_actions,last=(i == n_agent-1),deterministic=deterministic)
                # covert_time = time.time() - start_time
                # print("decode time: ",decod_time,"convert_time",covert_time)
                output_action[:, i, :] = action.unsqueeze(-1)
                output_action_log[:, i, :] = action_log.unsqueeze(-1)
                if i + 1 < n_agent:
                    shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)
            else:
                act_mean = decoder(shifted_action, obs_rep, obs)[:, i, :] 
                action_std = torch.sigmoid(decoder.log_std) * 0.5
                action,action_log = continuous_action(act_mean=act_mean,action_std=action_std,deterministic=deterministic)
                output_action[:, i, :] = action[:,1].reshape(-1,1)
                output_action_log[:, i, :] = action_log[:,1].reshape(-1,1)
                if i + 1 < n_agent:
                    shifted_action[:, i + 1, :] = action 
                
    return output_action, output_action_log

# To be modified
def semi_discrete_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv,available_actions=None,semi_index= -1):
    one_hot_action = F.one_hot(action[:,:semi_index,:].to(torch.long).squeeze(-1), num_classes=action_dim)
    continue_action = torch.broadcast_to(action[:,semi_index:,:],(action[:,semi_index:,:].shape[0],action[:,semi_index:,:].shape[1],action_dim))
    # print()
    action_all = torch.cat((one_hot_action,continue_action),1)
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    shifted_action[:, 1:, 1:] = action_all[:, :-1, :]
    logit = decoder(shifted_action, obs_rep, obs)
    act_mean = logit[:,semi_index:,:]
    logit = logit[:,:semi_index,:] 
    if available_actions is not None:
        logit[available_actions[:,:semi_index,:] == 0] = -1e10
        
    distri = Categorical(logits=logit)
    action_log_prev = distri.log_prob(action[:,:semi_index,:].squeeze(-1)).unsqueeze(-1)
    entropy_prev = distri.entropy().unsqueeze(-1)
    
    action_std = torch.sigmoid(decoder.log_std) * 0.5
    distri = Normal(act_mean, action_std)
    action_log_later = distri.log_prob(action[:,semi_index:,:])
    entropy_later = distri.entropy() 
    # print(action_log_prev.shape,entropy_prev.shape) 
    # print(action_log_later.shape,entropy_later.shape)
    action_log = torch.cat((action_log_prev,action_log_later[:,:,-1:]),1)
    entropy = torch.cat((entropy_prev,entropy_later[:,:,-1:]),1)
    return action_log, entropy

def discrete_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                available_actions=None, deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    if deterministic:
        starting_index = 0
        ending_index = 1
        continue_flag = True
        while ending_index <= n_agent and continue_flag:
            if ending_index == n_agent:
                continue_flag = False
            logits = decoder(shifted_action, obs_rep, obs)[:, starting_index:ending_index, :]
            for i in range(ending_index - starting_index):
                logit = logits[:,i,:]
                
                action,action_log = discrete_action(logit=logit,i=i+starting_index,available_actions=available_actions,last=False,deterministic=deterministic)
                output_action[:, i+starting_index, :] = action.unsqueeze(-1)
                output_action_log[:, i+starting_index, :] = action_log.unsqueeze(-1)
                if i + 1 < n_agent:
                    shifted_action[:, i + starting_index, 1:] = F.one_hot(action, num_classes=action_dim)
            if ending_index < n_agent:
                starting_index = ending_index
                ending_index +=4
                if ending_index > n_agent:
                    ending_index = n_agent
    else:
        for i in range(n_agent):
            logit = decoder(shifted_action, obs_rep, obs)[:, i, :]
            if available_actions is not None:
                logit[available_actions[:, i, :] == 0] = -1e10

            distri = Categorical(logits=logit)
            action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
            action_log = distri.log_prob(action)

            output_action[:, i, :] = action.unsqueeze(-1)
            output_action_log[:, i, :] = action_log.unsqueeze(-1)
            if i + 1 < n_agent:
                shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)
    return output_action, output_action_log


def discrete_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv,
                          available_actions=None):
    one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim)  # (batch, n_agent, action_dim)
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]
    logit = decoder(shifted_action, obs_rep, obs)
    if available_actions is not None:
        logit[available_actions == 0] = -1e10

    distri = Categorical(logits=logit)
    action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
    entropy = distri.entropy().unsqueeze(-1)
    return action_log, entropy


def continuous_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                  deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    output_action = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.float32)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent):
        act_mean = decoder(shifted_action, obs_rep, obs)[:, i, :]
        action_std = torch.sigmoid(decoder.log_std) * NORMAL_STD

        # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
        # distri = Normal(act_mean, log_std.exp())
        distri = Normal(act_mean, action_std)
        action = act_mean if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action
        output_action_log[:, i, :] = action_log
        if i + 1 < n_agent:
            shifted_action[:, i + 1, :] = action

        # print("act_mean: ", act_mean)
        # print("action: ", action)

    return output_action, output_action_log


def continuous_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    shifted_action[:, 1:, :] = action[:, :-1, :]

    act_mean = decoder(shifted_action, obs_rep, obs)
    action_std = torch.sigmoid(decoder.log_std) * NORMAL_STD
    distri = Normal(act_mean, action_std)

    # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
    # distri = Normal(act_mean, log_std.exp())

    action_log = distri.log_prob(action)
    entropy = distri.entropy()
    return action_log, entropy

def available_continuous_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                available_actions=None ,deterministic=True):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    output_action = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.float32)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    

    for i in range(n_agent):
        act_mean = decoder(shifted_action, obs_rep, obs)[:, i, :]
        action_std = torch.sigmoid(decoder.log_std)*NORMAL_STD
        # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
        # distri = Normal(act_mean, log_std.exp())
        
        distri = Normal(act_mean, action_std)
        action = act_mean if deterministic else distri.sample()
        action_log = distri.log_prob(action)
        if available_actions is not None:
            ava_a = available_actions[:, i, :]
            # In continous action space, the unavailable workers can only perform 0.
            # Since this action 0 is deterministic, the log_prob is set to 0 indicating probability = 1.
            action[ava_a == 0] = 0
            action_log[ava_a == 0] = 0

        output_action[:, i, :] = action
        output_action_log[:, i, :] = action_log
        if i + 1 < n_agent:
            shifted_action[:, i + 1, :] = action

        # print("act_mean: ", act_mean)
        # print("action: ", action)

    return output_action, output_action_log

def available_continuous_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv,available_actions=None):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    shifted_action[:, 1:, :] = action[:, :-1, :]

    act_mean = decoder(shifted_action, obs_rep, obs)
    action_std = torch.sigmoid(decoder.log_std)*NORMAL_STD
    distri = Normal(act_mean, action_std)

    # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
    # distri = Normal(act_mean, log_std.exp())
    
    # Generate log_prob normally
    action_log = distri.log_prob(action)
    # Setting log_prob to extremely small negative value to present the low prob of generated action
    # if available_actions is not None:
        # action_log[available_actions == 0] = -1e10
    entropy = distri.entropy()
    return action_log, entropy