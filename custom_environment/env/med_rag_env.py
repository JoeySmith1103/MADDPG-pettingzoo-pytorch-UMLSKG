# noqa
"""
# Simple Spread

```{figure} mpe_simple_spread.gif
:width: 140px
:name: simple_spread
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.mpe import simple_spread_v3` |
|----------------------|-----------------------------------------------|
| Actions              | Discrete/Continuous                           |
| Parallel API         | Yes                                           |
| Manual Control       | No                                            |
| Agents               | `agents= [agent_0, agent_1, agent_2]`         |
| Agents               | 3                                             |
| Action Shape         | (5)                                           |
| Action Values        | Discrete(5)/Box(0.0, 1.0, (5))                |
| Observation Shape    | (18)                                          |
| Observation Values   | (-inf,inf)                                    |
| State Shape          | (54,)                                         |
| State Values         | (-inf,inf)                                    |


This environment has N agents, N landmarks (default N=3). At a high level, agents must learn to cover all the landmarks while avoiding collisions.

More specifically, all agents are globally rewarded based on how far the closest agent is to each landmark (sum of the minimum distances). Locally, the agents are penalized if they collide with other agents (-1 for each collision). The relative weights of these rewards can be controlled with the
`local_ratio` parameter.

Agent observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_spread_v3.env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False)
```



`N`:  number of agents and landmarks

`local_ratio`:  Weight applied to local reward and global reward. Global reward weight will always be 1 - local reward weight.

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=80,
        local_ratio=0.5,
        max_cycles=25,
        continuous_actions=False,
    ):
        EzPickle.__init__(
            self,
            N=N,
            local_ratio=local_ratio,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(N)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
        )
        self.metadata["name"] = "simple_spread_v3"



env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    # DONE: modify the agent number
    # DONE: delete all lanmark settings
    def __init__(self):
        self.driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "admin"))
        self.tokenizer = AutoTokenizer.from_pretrained("JoeySmith1103/sapbert_2023ABFull")
        self.model = AutoModel.from_pretrained("JoeySmith1103/sapbert_2023ABFull").cuda()

    def close(self):
        self.driver.close()

    def get_embeddings(self, names, tokenizer, model, batch_size=128):
        all_embs = []
        for i in range(0, len(names), batch_size):
            toks = tokenizer.batch_encode_plus(names[i:i + batch_size],
                                            padding="max_length",
                                            max_length=25,
                                            truncation=True,
                                            return_tensors="pt")
            toks_cuda = {k: v.cuda() for k, v in toks.items()}
            cls_rep = model(**toks_cuda)[0][:, 0, :]
            all_embs.append(cls_rep.cpu().detach().numpy())
        all_embs = np.concatenate(all_embs, axis=0)
        return all_embs

    def calculate_average_similarity_among_neighbors(self, node):
        neighbors = self.find_one_hop_neighbors(node_id=node)
        if not neighbors:
            return {"num_neighbors": 0, "avg_similarity": 0.0, "variance_similarity": 0.0}

        neighbor_names = [record['neighbor_name'] for record in neighbors]
        embeddings = self.get_embeddings(neighbor_names, self.tokenizer, self.model)

        similarity_matrix = cosine_similarity(embeddings)
        similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

        avg_similarity = np.mean(similarities)
        variance_similarity = np.var(similarities)

        return {
            "num_neighbors": len(neighbor_names),
            "avg_similarity": avg_similarity,
            "variance_similarity": variance_similarity,
        }
    
    def find_one_hop_neighbors(self, node_id):
        query = """
        MATCH (c:Concept {CUI: $cui})-[:PAR|:CHD|:RN|:AQ|:SY*1]->(neighbor:Concept)
        RETURN neighbor.CUI AS neighbor_cui, neighbor.name AS neighbor_name
        """
        with self.driver.session() as session:
            result = session.run(query, cui=node_id)
            neighbors = [{"neighbor_cui": record["neighbor_cui"], "neighbor_name": record["neighbor_name"]} for record in result]
            return neighbors
        
    def make_world(self, N=3):
        world = World()
        # set any world properties first
        world.dim_c = 0 # prohabited communication
        num_agents = N
        # num_landmarks = N
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            # 2025 / 1 / 16
            # agent.state.p_vel = np.zeros(world.dim_p)
            # agent.state.c = np.zeros(world.dim_c)
        ## 2025 / 1 / 23 test, should add data about the agent
        ## TODO

    # def benchmark_data(self, agent, world):
    #     rew = 0
    #     collisions = 0
    #     occupied_landmarks = 0
    #     min_dists = 0
    #     for lm in world.landmarks:
    #         dists = [
    #             np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
    #             for a in world.agents
    #         ]
    #         min_dists += min(dists)
    #         rew -= min(dists)
    #         if min(dists) < 0.1:
    #             occupied_landmarks += 1
    #     if agent.collide:
    #         for a in world.agents:
    #             if self.is_collision(a, agent):
    #                 rew -= 1
    #                 collisions += 1
    #     return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        # rew = 0
        # if agent.collide:
        #     for a in world.agents:
        #         rew -= 1.0 * (self.is_collision(a, agent) and a != agent)
        # return rew
        
        ## 2025 / 1 / 23 test
        similarity_stats = self.calculate_similarity(agent.node_id)
        avg_similarity = similarity_stats["avg_similarity"]

        # reward 是平均相似度
        return avg_similarity

    def global_reward(self, world):
        rew = 0
        # for lm in world.landmarks:
        #     dists = [
        #         np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
        #         for a in world.agents
        #     ]
        #     rew -= min(dists)
        return rew

    def observation(self, agent, world):
        # # get positions of all entities in this agent's reference frame
        # entity_pos = []
        # # for entity in world.landmarks:  # world.entities:
        # #     entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # # communication of all other agents
        # comm = []
        # other_pos = []
        # for other in world.agents:
        #     if other is agent:
        #         continue
        #     comm.append(other.state.c)
        #     other_pos.append(other.state.p_pos - agent.state.p_pos)
        # return np.concatenate(
        #     [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm
        # )

        ## 2025 / 1 / 23 test
        similarity_stats = self.calculate_average_similarity_among_neighbors(agent.node_id)
        observation = np.concatenate([
            [similarity_stats["num_neighbors"]],
            [similarity_stats["avg_similarity"]],
            [similarity_stats["variance_similarity"]]
        ])
        return observation
    

test = Scenario()
print(test.find_one_hop_neighbors("C0085131"))
print(test.calculate_average_similarity_among_neighbors("C0085131"))