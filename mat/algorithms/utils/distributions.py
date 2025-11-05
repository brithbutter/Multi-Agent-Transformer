import torch
import torch.nn as nn
from .util import init

"""
Modify standard PyTorch distributions so they to make compatible with this codebase. 
"""

#
# Standardize distribution interfaces
#
def init_method(use_orthogonal):
    return [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
def init_(m,use_orthogonal,gain): 
    return init(m, init_method(use_orthogonal), lambda x: nn.init.constant_(x, 0), gain)

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

class FixedOneHotCategorical(torch.distributions.OneHotCategorical):
    def sample(self):
        return super().sample()
    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )
    def mode(self):
        return super().mode()
# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions)
        # return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()

        self.linear = init_(nn.Linear(num_inputs, num_outputs), use_orthogonal, gain)

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


# class DiagGaussian(nn.Module):
#     def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
#         super(DiagGaussian, self).__init__()
#
#         init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
#         def init_(m):
#             return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
#
#         self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
#         self.logstd = AddBias(torch.zeros(num_outputs))
#
#     def forward(self, x, available_actions=None):
#         action_mean = self.fc_mean(x)
#
#         #  An ugly hack for my KFAC implementation.
#         zeros = torch.zeros(action_mean.size())
#         if x.is_cuda:
#             zeros = zeros.cuda()
#
#         action_logstd = self.logstd(zeros)
#         return FixedNormal(action_mean, action_logstd.exp())

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01, args=None):
        super(DiagGaussian, self).__init__()

        if args is not None:
            self.std_x_coef = args.std_x_coef
            self.std_y_coef = args.std_y_coef
        else:
            self.std_x_coef = 1.
            self.std_y_coef = 0.5
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs), use_orthogonal, gain)
        log_std = torch.ones(num_outputs) * self.std_x_coef
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, x, available_actions=None):
        action_mean = self.fc_mean(x)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        return FixedNormal(action_mean, action_std)

class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        
        self.linear = init_(nn.Linear(num_inputs, num_outputs),use_orthogonal, gain)

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

# Mixed Distribution
class MixedCategoricalDiagGaussianDistribution(nn.Module):
    def __init__(self, num_inputs, num_discrete_outputs,num_continuous_outputs, use_orthogonal=True, gain=0.01, args=None):
        super(MixedCategoricalDiagGaussianDistribution, self).__init__()
        self.num_discrete_outputs = num_discrete_outputs
        self.num_continuous_outputs = num_continuous_outputs
        if args is not None:
            self.std_x_coef = args.std_x_coef
            self.std_y_coef = args.std_y_coef
        else:
            self.std_x_coef = 1.
            self.std_y_coef = 0.5 
        self.linear = init_(nn.Linear(num_inputs, num_discrete_outputs+num_continuous_outputs), use_orthogonal, gain)
        log_std = torch.ones(num_continuous_outputs) * self.std_x_coef
        self.log_std = torch.nn.Parameter(log_std)
    def forward(self, x, available_actions=None):
        output = self.linear(x)
        discrete_logits = output[:, :self.num_discrete_outputs]
        continuous_mean = output[:, self.num_discrete_outputs:self.num_discrete_outputs+self.num_continuous_outputs]
        if available_actions is not None:
            discrete_logits[available_actions[:,:self.num_discrete_outputs] == 0] = -1e10
        discrete_action_distri = FixedOneHotCategorical(logits=discrete_logits)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef 
        continuous_action_distri = FixedNormal(continuous_mean, action_std)
        return discrete_action_distri, continuous_action_distri
class MultiMixedCategoricalDiagGaussianDistribution(nn.Module):
    def __init__(self, num_inputs, num_discrete_outputs,num_continuous_outputs, n_components,use_orthogonal=True, gain=0.01, args=None):
        super(MultiMixedCategoricalDiagGaussianDistribution, self).__init__()
        self.num_discrete_outputs = num_discrete_outputs
        self.num_continuous_outputs = num_continuous_outputs
        self.n_components = n_components
        if args is not None:
            self.std_x_coef = args.std_x_coef
            self.std_y_coef = args.std_y_coef
        else:
            self.std_x_coef = 1.
            self.std_y_coef = 0.5 
        self.linear = init_(nn.Linear(num_inputs, (num_discrete_outputs+num_continuous_outputs)*n_components), use_orthogonal, gain)
        log_stds = torch.ones(n_components,num_continuous_outputs) * self.std_x_coef
        self.log_stds = torch.nn.Parameter(log_stds)
    def forward(self, x, available_actions=None):
        output = self.linear(x).reshape(x.shape[0],self.n_components,-1)
        if available_actions is not None:
            output[available_actions == 0] = -1e10
        discrete_action_distris = []
        continuous_action_distris = []
        for i in range(self.n_components):
            discrete_logits = output[:, i, :self.num_discrete_outputs]
            continuous_mean = output[:, i, self.num_discrete_outputs:self.num_discrete_outputs+self.num_continuous_outputs]
            discrete_action_distris.append(FixedOneHotCategorical(logits=discrete_logits))
            action_std = torch.sigmoid(self.log_stds[i] / self.std_x_coef) * self.std_y_coef 
            continuous_action_distris.append(FixedNormal(continuous_mean, action_std))
        return discrete_action_distris, continuous_action_distris