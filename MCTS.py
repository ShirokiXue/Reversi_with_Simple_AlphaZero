from math import sqrt, log
import numpy as np
import random
import torch

class Node:
    def __init__(self, env, parent=None):
        self.env = env
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.Q_value = 0
        self.possible_actions = env.get_possible_action()
        # self.action_probs = None
        # self.choose_action = None

    # def select(self) -> "Node":
    #     not_visits = [v for v in self.children if v.visits == 0]
    #     if not_visits:
    #         return np.random.choice(not_visits)
    #     p = [v.wins / v.visits + 0.5*sqrt(2 * log(self.visits) / 1+v.visits) for v in self.children]
    #     p = np.array(p) / np.sum(p)
    #     r = np.random.choice(range(len(self.children)), p=p)
    #     return self.children[r]

    def predict(self, nnet):
        preprocessed_data = nnet.preprocess(self.env.board, self.env.turn)
        self.Q_value = nnet(preprocessed_data).item()
        if self.Q_value < -1: self.Q_value = -1
        if self.Q_value > 1:  self.Q_value = 1 

    def expand(self):
        if not self.children:
            for action in self.possible_actions:
                clone_env = self.env.clone()
                clone_env.step(action)
                self.children[action] = Node(clone_env, self) 

    def pick_best_action_from_child(self):
        max_u, best_a = -float("inf"), -1
        for a in self.possible_actions:
            u = self.children[a].Q_value*self.children[a].env.turn*self.env.turn
            if u>max_u:
                max_u = u
                best_a = a
        action = best_a
        return action

    def pick_random_action_according_to_weights(self, c_puct=1):
        actions, weights = [], []
        for a in self.possible_actions:
            q = self.children[a].Q_value*self.children[a].env.turn*self.env.turn
            u = (q+1)/2 + 0.00001 + c_puct*sqrt(self.visits)/(self.children[a].visits+1)
            actions.append(a)
            weights.append(u)
        action = random.choices(actions, weights=weights)[0]
        return action

    def pick_random_action(self):
        action = random.choice(self.possible_actions)
        return action

    def backpropagate(self, v):
        cur: Node = self
        while cur:
            if cur.env.get_turn() == self.env.get_turn():
                cur.update(v)
            else:
                cur.update(-v)
            cur = cur.parent

    def update(self, v):
        self.Q_value = (self.visits*self.Q_value + v)/(self.visits+1)
        self.visits += 1 

class MCTS():
    def __init__(self, env, nnet) -> None:
        self.env = env
        self.root = Node(env=env.clone(), parent=None)
        self.first_node = self.root
        self.nnet = nnet
        self.nnet.to("cuda")

    def explore(self, node: Node, c_puct=1) -> None:
        if node.env.isEnd:
            node.Q_value = node.env.get_result()*node.env.get_turn()
            node.backpropagate(node.Q_value)
            return

        if node.visits == 0:
            node.visits += 1
            node.predict(self.nnet)
            node.backpropagate(node.Q_value)
            node.expand()
            return
        
        action = node.pick_random_action_according_to_weights(c_puct=1)
        self.explore(node.children[action])
        return

    def step(self, action):
        self.root.expand()
        self.root = self.root.children[action]