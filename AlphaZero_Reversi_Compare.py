import threading
from MCTS import MCTS
from model.v4.NNet import *
from Reversi import *

from copy import deepcopy
from tqdm import tqdm

from utils import *

class ai_player():
    def __init__(self, env:Reversi, nnet:NNet):
        self.env  = env
        self.nnet = nnet
        self.mcts = MCTS(env, self.nnet) 

    def intelligent_action(self, numMCTSSims=30, c_puct=1):
        for _ in range(numMCTSSims):
            self.mcts.explore(self.mcts.root, c_puct=c_puct)
        a = self.mcts.root.pick_random_action_according_to_weights(c_puct=c_puct)
        return a

    def best_action(self, numMCTSSims=30, c_puct=1):
        for _ in range(numMCTSSims):
            self.mcts.explore(self.mcts.root, c_puct=c_puct)
        a = self.mcts.root.pick_best_action_from_child()
        return a

    def random_action(self):
        possible_acitons = self.env.get_possible_action()
        return random.choice(list(possible_acitons))

    def step(self, action):
        self.mcts.step(action)

class CompareWorker(threading.Thread):
    def __init__(self, new_nnet, original_nnet, score:list, switch=1):
        threading.Thread.__init__(self)
        self.new_nnet = new_nnet
        self.original_nnet = original_nnet 
        self.score = score
        self.switch = switch

    def run(self):
        env = Reversi()
        env.reset()
        old_player = ai_player(env=env, nnet=self.original_nnet)
        new_player = ai_player(env=env, nnet=self.new_nnet)

        for _ in range(60):
            if env.isEnd: break
            turn = env.get_turn()
            if turn == self.switch:
                action = old_player.best_action(numMCTSSims=10, c_puct=0)
                action = old_player.random_action()
            elif turn == -self.switch:
                action = new_player.best_action(numMCTSSims=50, c_puct=0)
                # action = new_player.random_action()
            env.step(action)
            old_player.step(action)
            new_player.step(action)

        result = env.get_result()
        if result == -1*self.switch:
            self.score.append("N")
        else: 
            self.score.append("O")
def NN_compare(score, new_nnet, original_nnet, n_games) -> NNet:
    
    workers =  [CompareWorker(new_nnet, original_nnet, score) for _ in range(n_games)]
    workers += [CompareWorker(new_nnet, original_nnet, score, switch=-1) for _ in range(n_games)]
    for w in tqdm(workers): 
        w.start()
        w.join()
        

    if len(score) != 0:
        win_rate = score.count("N")/len(score)
    else:
        return -1

    print(f"result -> new_player({score.count('N')}):old_player({score.count('O')}), win_rate: {win_rate}")
    del workers
    return win_rate 
    
if __name__ == "__main__":

    threshold=0.55
    while(True):
        score = []
        old_nnet = NNet()
        old_nnet.load_latest_model()
        new_nnet = NNet()
        new_nnet.load_latest_model()
        print("\nComparing...")
        frac_win = NN_compare(score, new_nnet, old_nnet, n_games = 50)
        # if frac_win > threshold: 
        #     new_nnet.save_model()
        break