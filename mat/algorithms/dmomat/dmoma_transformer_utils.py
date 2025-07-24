import torch
import torch.nn as nn
from mat_src.mat.algorithms.utils.transformer_utils import init_, EncodeBlock

OPTION_MULTI_OBJECTIVE_APPROACH = 1
class DMOEncoder(nn.Module):

    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state,n_objective = 2,multi_objective_approach=OPTION_MULTI_OBJECTIVE_APPROACH):
        super(DMOEncoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state
        self.multi_objective_approach = multi_objective_approach
        # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))

        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate='relu'), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate='relu'), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        
        
        if self.multi_objective_approach == 0:
            self.heads = nn.ModuleList()
            for i_objective in range(n_objective):
                self.heads.append(nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate='relu'), nn.GELU(), nn.LayerNorm(n_embd),
                                    init_(nn.Linear(n_embd, 1))))
        elif self.multi_objective_approach == 1:
            self.heads = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate='relu'), nn.GELU(), nn.LayerNorm(n_embd),
                                    init_(nn.Linear(n_embd, n_objective)))

    def forward(self, state, obs):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        if self.encode_state:
            state_embeddings = self.state_encoder(state)
            x = state_embeddings
        else:
            obs_embeddings = self.obs_encoder(obs)
            x = obs_embeddings

        rep = self.blocks(self.ln(x))
        if OPTION_MULTI_OBJECTIVE_APPROACH == 0:
            v_locs = []
            for head in self.heads:
                v_locs.append(head(rep))
            v_locs = torch.cat(v_locs, dim=-1)
        elif OPTION_MULTI_OBJECTIVE_APPROACH == 1:
            v_locs = self.heads(rep)
        # v_locs = torch.stack(v_locs,2)
        return v_locs, rep