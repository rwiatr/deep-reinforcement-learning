import torch
import torch.nn as nn
import torch.optim as optim

from drl.network.body import FCNet, vanilla_action_fc_net
import torch.nn.functional as F


def vanilla_acn(s_dim, a_dim, lr_a=None, lr_c=None, wd_a=0, wd_c=0, seed=None):
    actor_net = FCNet(l_dims=(s_dim, 300, 150, a_dim), actv=(F.relu, F.relu, F.tanh), seed=seed)
    critic_net = vanilla_action_fc_net(s_dim, a_dim, seed=seed)
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
