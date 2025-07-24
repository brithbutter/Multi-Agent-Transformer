import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
import time
from torch.distributions import Categorical
from mat.algorithms.utils.util import check, init
from mat.algorithms.utils.transformer_act import *
from mat_src.mat.algorithms.utils.transformer_utils import Decoder
from mat_src.mat.algorithms.dmomat.dmoma_transformer_utils import DMOEncoder 

OPTION_MULTI_OBJECTIVE_APPROACH = 1



class DMOMultiAgentTransformer(nn.Module):

    def __init__(self, state_dim, obs_dim, action_dim, n_agent,
                n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                action_type='Discrete', dec_actor=False, share_actor=False,semi_index= -1,n_objective = 2):
        super(DMOMultiAgentTransformer, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device
        self.semi_index = semi_index
        self.n_objective = n_objective
        # state unused
        # state_dim = 37
        
        self.encoder = DMOEncoder(state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state,self.n_objective,multi_objective_approach=OPTION_MULTI_OBJECTIVE_APPROACH)
        self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                            self.action_type, dec_actor=dec_actor, share_actor=share_actor)
        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)

    def forward(self, state, obs, action, available_actions=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        # state unused
        # ori_shape = np.shape(state)
        # state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(state)[0]
        v_locs, obs_rep = self.encoder(state, obs)
        if self.action_type == 'Discrete':
            action = action.long()
            action_log, entropy = discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                        self.n_agent, self.action_dim, self.tpdv, available_actions)
        elif self.action_type == "Semi_Discrete":
            action_log, entropy = semi_discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                        self.n_agent, self.action_dim, self.tpdv, available_actions,semi_index=self.semi_index)
        elif self.action_type == "Continous":
            action_log, entropy = continuous_parallel_act(self.decoder, obs_rep, obs, action, batch_size,self.n_agent, self.action_dim, self.tpdv)
        elif self.action_type == "Available_Continous":
            action_log, entropy = available_continuous_parallel_act(self.decoder, obs_rep, obs, action, batch_size,self.n_agent, self.action_dim, self.tpdv, available_actions=available_actions)

        return action_log, v_locs, entropy

    def get_actions(self, state, obs, available_actions=None, deterministic=False, stride= 2):
        # state unused
        # ori_shape = np.shape(obs)
        # state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(obs)[0]
        v_locs, obs_rep = self.encoder(state, obs)
        
        if self.action_type == "Discrete":
            output_action, output_action_log = discrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                            self.n_agent, self.action_dim, self.tpdv,
                                                                            available_actions, deterministic)
        elif self.action_type == "Semi_Discrete":
            output_action, output_action_log = semi_discrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                                self.n_agent, self.action_dim, self.tpdv,
                                                                                available_actions, deterministic,semi_index=self.semi_index,stride=stride)
            
        elif self.action_type == "Continous":
            output_action, output_action_log = continuous_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                            self.n_agent, self.action_dim, self.tpdv,
                                                                            deterministic)
        elif self.action_type == "Available_Continous":
            output_action, output_action_log = available_continuous_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                            self.n_agent, self.action_dim, self.tpdv,
                                                            available_actions,deterministic)
        else:
            raise NotImplementedError
        return output_action, output_action_log, v_locs

    def get_values(self, state, obs, available_actions=None):
        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        v_tots, obs_rep = self.encoder(state, obs)
        return v_tots



