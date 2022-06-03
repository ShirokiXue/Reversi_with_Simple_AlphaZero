from model.v4.NNet import NNet
from Reversi import *
import random

if __name__ == "__main__":

    env = Reversi()
    net = NNet()
    net.load_latest_model()
    net.to("cuda")
    
    for i in range(60):
        if env.isEnd: break
        actions = env.get_possible_action()
        a = random.choice(actions)
        env.step(a)
        env.show_board()
        print(f"Turn = {env.get_turn()}")
        print(net(net.preprocess(env.board, turn=env.turn)).item())