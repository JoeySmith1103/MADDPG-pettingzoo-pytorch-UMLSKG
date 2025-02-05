# from copy import deepcopy
# from typing import List

# import torch
# import torch.nn.functional as F
# from torch import nn, Tensor
# from torch.optim import Adam


# class Agent:
#     """Agent that can interact with environment from pettingzoo"""

#     def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr):
#         self.actor = MLPNetwork(obs_dim, act_dim)

#         # critic input all the observations and actions
#         # if there are 3 agents for example, the input for critic is (obs1, obs2, obs3, act1, act2, act3)
#         self.critic = MLPNetwork(global_obs_dim, 1)
#         self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
#         self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
#         self.target_actor = deepcopy(self.actor)
#         self.target_critic = deepcopy(self.critic)

#     @staticmethod
#     def gumbel_softmax(logits, tau=1.0, eps=1e-20):
#         # NOTE that there is a function like this implemented in PyTorch(torch.nn.functional.gumbel_softmax),
#         # but as mention in the doc, it may be removed in the future, so i implement it myself
#         epsilon = torch.rand_like(logits)
#         logits += -torch.log(-torch.log(epsilon + eps) + eps)
#         return F.softmax(logits / tau, dim=-1)

#     def action(self, obs, model_out=False):
#         # this method is called in the following two cases:
#         # a) interact with the environment
#         # b) calculate action when update actor, where input(obs) is sampled from replay buffer with size:
#         # torch.Size([batch_size, state_dim])

#         # logits = self.actor(obs)  # torch.Size([batch_size, action_size])
#         # # action = self.gumbel_softmax(logits)
#         # action = F.gumbel_softmax(logits, hard=True)
#         # if model_out:
#         #     return action, logits
#         # return action

#         ## 2025 / 1 / 23 test
#         logits = self.actor(obs)
#         action_probabilities = F.gumbel_softmax(logits, hard=True)
#         action = torch.argmax(action_probabilities).item()

#         if action == 0:  # 1-hop search
#             # 執行 1-hop search 查詢
#             self.execute_one_hop_search()
#         elif action == 1:  # 跨群 search
#             # 執行跨群搜索
#             self.execute_cross_group_search()
#         elif action == 2:  # 停止 search
#             pass

#         if model_out:
#             return action, logits
#         return action
    
#     def target_action(self, obs):
#         # when calculate target critic value in MADDPG,
#         # we use target actor to get next action given next states,
#         # which is sampled from replay buffer with size torch.Size([batch_size, state_dim])

#         logits = self.target_actor(obs)  # torch.Size([batch_size, action_size])
#         # action = self.gumbel_softmax(logits)
#         action = F.gumbel_softmax(logits, hard=True)
#         return action.squeeze(0).detach()

#     def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
#         x = torch.cat(state_list + act_list, 1)
#         return self.critic(x).squeeze(1)  # tensor with a given length

#     def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
#         x = torch.cat(state_list + act_list, 1)
#         return self.target_critic(x).squeeze(1)  # tensor with a given length

#     def update_actor(self, loss):
#         self.actor_optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
#         self.actor_optimizer.step()

#     def update_critic(self, loss):
#         self.critic_optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
#         self.critic_optimizer.step()


# class MLPNetwork(nn.Module):
#     def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
#         super(MLPNetwork, self).__init__()

#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             non_linear,
#             nn.Linear(hidden_dim, hidden_dim),
#             non_linear,
#             nn.Linear(hidden_dim, out_dim),
#         ).apply(self.init)

#     @staticmethod
#     def init(m):
#         """init parameter of the module"""
#         gain = nn.init.calculate_gain('relu')
#         if isinstance(m, nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight, gain=gain)
#             m.bias.data.fill_(0.01)

#     def forward(self, x):
#         return self.net(x)

from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam

from pettingzoo.mpe._mpe_utils.core import Agent as PZAgent


class CustomAgent(PZAgent):
    """Agent that interacts with the environment from PettingZoo"""

    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.global_obs_dim = global_obs_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.actor = MLPNetwork(obs_dim, act_dim)
        self.critic = MLPNetwork(global_obs_dim, 1)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

        # state initialization
        self.state = {
            "num_neighbors": 0,
            "avg_similarity": 0.0,
            "variance_similarity": 0.0
        }
        self.node_id = None  # 初始 CUI

    def update_state(self, scenario):
        """使用 scenario 的 `calculate_average_similarity_among_neighbors()` 更新 `state`"""
        similarity_stats = scenario.graph_handler.calculate_average_similarity_among_neighbors(self.node_id)
        self.state["num_neighbors"] = similarity_stats["num_neighbors"]
        self.state["avg_similarity"] = similarity_stats["avg_similarity"]
        self.state["variance_similarity"] = similarity_stats["variance_similarity"]

    def action(self, obs, model_out=False):
        """Agent 根據 observation 選擇行動"""
        logits = self.actor(obs)
        action_probabilities = F.gumbel_softmax(logits, hard=True)
        action = torch.argmax(action_probabilities).item()
        print(f"Action: {action}")
        if model_out:
            return action, logits
        return action

    def execute_action(self, action, scenario):
        """執行對應的動作並更新 `state`"""
        if action == 0:  # 1-hop search
            print(f"Executing action {action} on node {self.node_id}")
            neighbors = scenario.graph_handler.find_one_hop_neighbors(self.node_id)
            if neighbors:
                self.node_id = neighbors[0]['neighbor_cui']
                print(f"Found {len(neighbors)} neighbors, moving to {self.node_id}")

        elif action == 1:  # 跨群 search
            pass  # TODO: Implement跨群搜尋邏輯

        elif action == 2:  # 停止 search
            pass

        self.update_state(scenario)  # ✅ 執行動作後更新 `state`

    def target_action(self, obs):
        logits = self.target_actor(obs)
        action = F.gumbel_softmax(logits, hard=True)
        return action.squeeze(0).detach()

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.critic(x).squeeze(1)

    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic(x).squeeze(1)

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()


class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(MLPNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        """初始化網路參數"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)