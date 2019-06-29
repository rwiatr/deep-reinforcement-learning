import torch
import torch.nn as nn
import torch.optim as optim

from drl.network.body import FCNet, ActionFCNet
import torch.nn.functional as F


def acn(a_dim, a_l_dims, c_l_dims, lr_a=None, lr_c=None, wd_a=0, wd_c=0, seed=None):
    a_actv = list([F.relu for _ in range(len(a_l_dims) - 2)]) + [F.tanh]
    c_actv = list([F.relu for _ in range(len(c_l_dims) - 2)]) + [None]

    actor_net = FCNet(l_dims=a_l_dims, actv=a_actv, seed=seed)
    critic_net = ActionFCNet(l_dims=c_l_dims, actv=c_actv, a_dim=a_dim, action_cat=1, seed=seed)
    actor_opt = None if lr_a is None else optim.Adam(actor_net.parameters(), lr=lr_a, weight_decay=wd_a)
    critic_opt = None if lr_c is None else optim.Adam(critic_net.parameters(), lr=lr_c, weight_decay=wd_c)

    return ActorCriticNet(
        actor_net=actor_net,
        critic_net=critic_net,
        actor_opt=actor_opt,
        critic_opt=critic_opt)


def default_acn(s_dim, a_dim, lr_a=None, lr_c=None, wd_a=0, wd_c=0, seed=None):
    actor_net = FCNet(l_dims=(s_dim, 400, 300, a_dim), actv=(F.relu, F.relu, F.tanh), seed=seed)
    critic_net = ActionFCNet(l_dims=(s_dim, 400, 300, 1), actv=(F.relu, F.relu, None),
                             a_dim=a_dim, action_cat=1, seed=seed)
    actor_opt = None if lr_a is None else optim.Adam(actor_net.parameters(), lr=lr_a, weight_decay=wd_a)
    critic_opt = None if lr_c is None else optim.Adam(critic_net.parameters(), lr=lr_c, weight_decay=wd_c)

    return ActorCriticNet(
        actor_net=actor_net,
        critic_net=critic_net,
        actor_opt=actor_opt,
        critic_opt=critic_opt)


class ActorCriticNet(nn.Module):

    def __init__(self, actor_net, critic_net, actor_opt=None, critic_opt=None, seed=None):
        super(ActorCriticNet, self).__init__()
        if seed:
            self.seed = torch.manual_seed(seed)
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.actor_opt = actor_opt
        self.critic_opt = critic_opt

    def forward(self, state):
        return self.actor(state)

    def actor(self, state):
        return self.actor_net(state)

    def critic(self, state, action):
        return self.critic_net(state, action)

    def learn_critic(self, loss):
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()

    def learn_actor(self, loss):
        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()

    def update(self, other, tau):
        for target_param, local_param in zip(self.parameters(), other.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
