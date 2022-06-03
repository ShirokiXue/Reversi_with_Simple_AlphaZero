from numpy import blackman
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical
from datetime import datetime
import glob

LEARNING_RATE = 0.00001

class NNet(nn.Module):

    def __init__(self):
        
        super(NNet, self).__init__()

        self.con1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.con2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.con3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(259, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
        self.optimizer  = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def forward(self, data):
        x1 = data[0]
        x1 = torch.tanh(self.con1(x1))
        x1 = torch.tanh(self.con2(x1))
        x1 = torch.tanh(self.con3(x1))

        x1 = x1.view(1, -1)
        x2 = data[1].view(1, -1)
        x = torch.cat([x1, x2], 1)

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)

        return x

    def preprocess(self, board, turn):

        board_a = [[float(c==turn) for c in r] for r in board]
        board_b = [[float(c==-turn) for c in r] for r in board]
        board = [board_a, board_b]
        
        a_count = sum([sum([int(c==turn) for c in r]) for r in board])
        b_count = sum([sum([int(c==-turn) for c in r]) for r in board])
        total = a_count + b_count
        count = torch.tensor([a_count, b_count, total]).to("cuda")
        
        board = torch.tensor(board).to("cuda")

        return [board, count]

    def save_model(self, filename=None):
        if not filename:
            time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            torch.save(self.state_dict(), f"./model/v4/model_weights_v4_{time}.pth")
        else:
            torch.save(self.state_dict(), f"./model/v4/{filename}.pth")

    def load_model(self, filename):
        files = glob.glob(f".\\model\\v4\\{filename}.pth")
        if files:
            file = files[0]
            self.load_state_dict(torch.load(file))
            print(f"Load file {file}")
        else:
            print("No file can be loaded.")

    def load_latest_model(self):
        files = glob.glob(".\\model\\v4\\*.pth")

        if files:
            file = files[-1]
            self.load_state_dict(torch.load(file))
            print(f"Load file {file}")
        else:
            print("No file can be loaded.")

    def load_first_model(self):
        files = glob.glob(".\\model\\v4\\*.pth")

        if files:
            file = files[0]
            self.load_state_dict(torch.load(file))
            print(f"Load file {file}")
        else:
            print("No file can be loaded.")

if __name__ == "__main__":
    
    D = ['A','B','C','D','E','F','G','H']
    BLACK = 1
    WHITE = -1
    def pos_encode(input: int):
        return f"{(input//8+1)}{D[input%8]}"

    net = NNet()
    # net.load_model(filename="pre-train_model_v4")
    net.load_first_model()
    net.to("cuda")
    print(net)

    a = [[0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,-1,1,0,0,0],
        [0,0,0,1,-1,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0]]
    print(net(net.preprocess(a, turn=BLACK)).item())
    print(net(net.preprocess(a, turn=WHITE)).item())
    print()


    b = [[1,1,1,1,1,1,1,1],
        [1,-1,1,-1,1,-1,1,1],
        [1,1,-1,1,-1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,-1,-1,1,1,1],
        [1,-1,1,-1,1,1,1,1],
        [1,1,1,1,-1,1,1,1],
        [1,1,1,-1,-1,-1,-1,-1]]
    print(net(net.preprocess(b, turn=BLACK)).item())
    print(net(net.preprocess(b, turn=WHITE)).item())
    print()