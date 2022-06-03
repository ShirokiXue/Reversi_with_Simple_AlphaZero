import threading
from MCTS import MCTS
from model.v4.NNet import *
from Reversi import *

from tqdm import tqdm
import torch

from utils import *

class SampleWorker(threading.Thread):
    def __init__(self, memory:list, nnet, numMCTSSims, c_puct):
        threading.Thread.__init__(self)
        self.memory      = memory
        self.nnet        = nnet
        self.nnet.to("cuda")
        self.numMCTSSims = numMCTSSims
        self.c_puct      = c_puct

    def run(self):
        
        env = Reversi(render=False)
        mcts = MCTS(env, self.nnet)                                           # initialise search tree
        
        for _ in tqdm(range(60)):
                if env.isEnd: break
                for _ in range(self.numMCTSSims):
                    mcts.explore(mcts.root) 
                a = mcts.root.pick_random_action_according_to_weights(c_puct=self.c_puct)
                env.step(a)
                mcts.step(a)
                # buffer.append([mcts.root, None])              # rewards can not be determined yet

        buffer = []   
        get_all_nodes(buffer, mcts.first_node)
        self.memory.extend(buffer)

def trainNNet(nnet: NNet, memory):
    batch_size = 1024

    for i in tqdm(range(len(memory)//batch_size+1)):
        buffer = memory[i*batch_size:(i+1)*batch_size]
        value_losses  = []
        for node in buffer:    

            reward = node.Q_value
            Q_value = nnet(nnet.preprocess(node.env.board, node.env.turn))
            value_losses.append(F.smooth_l1_loss(Q_value[0], torch.tensor([reward]).to("cuda")))

            # reward = -node.Q_value
            # Q_value = nnet(nnet.preprocess(node.env.board, -node.env.turn))
            # value_losses.append(F.smooth_l1_loss(Q_value[0], torch.tensor([reward]).to("cuda")))

        nnet.optimizer.zero_grad()
        loss = torch.stack(value_losses).sum()
        loss.backward()
        nnet.optimizer.step()
    return nnet

def policyIterSP(numIters=1, numEps=3, threshold=0.6, numMCTSSims=300, c_puct=0) -> NNet:
    nnet = NNet()
    nnet.to("cuda")
    nnet.load_first_model()
    i = 0
    while(True):
        memory = []
        i += 1
        print(f"\nSampling... Iter: {i}")
        env = Reversi(render=False)
        mcts = MCTS(env, nnet)                                           # initialise search tree
        
        for _ in tqdm(range(60)):
            if env.isEnd: break
            for _ in range(numMCTSSims):
                mcts.explore(mcts.root) 
            a = mcts.root.pick_random_action_according_to_weights(c_puct=c_puct)
            env.step(a)
            mcts.step(a)

        buffer = []   
        # get_all_nodes(buffer, mcts.first_node)
        get_selected_nodes(buffer, mcts.root)
        memory.extend(buffer)
        print(f"Training NNet...")
        trainNNet(nnet, memory)
        del memory, env, mcts
        # del workers 
        nnet.save_model(filename="model_v4")
    
if __name__ == "__main__":
    policyIterSP()