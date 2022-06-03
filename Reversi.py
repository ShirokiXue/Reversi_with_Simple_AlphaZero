# -*- coding: utf-8 -*-
"""
Created on Fri May  6 02:06:25 2022

@author: xxasd
"""
from copy import deepcopy
import random

EMPTY = 0
BLACK = 1
WHITE = -1
B = [[0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,-1,1,0,0,0],
     [0,0,0,1,-1,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0]]

# B = [[0,0,0,0,0,0,0,0],
#      [0,0,0,0,0,0,0,0],
#      [0,0,0,0,0,0,0,0],
#      [0,0,0,1,-1,1,-1,0],
#      [0,0,0,0,0,0,0,0],
#      [0,0,0,0,0,0,0,0],
#      [0,0,0,0,0,0,0,0],
#      [0,0,0,0,0,0,0,0]]

def pos_decode(pos: str):
    r = int(pos[0]) - 1
    c = ord(pos[1]) - 64 - 1
    return r*8+c

class Reversi():
    
    def __init__(self, board = B, turn = 1, render=False):
        
        self.board = board
        self.turn = turn
        self.save_board = []
        self.save_turn  = []
        self.isEnd = (not self.get_possible_action(BLACK)) \
                    and (not self.get_possible_action(WHITE))
        self.render = render
        
    def show_board(self):
        print("\n  A B C D E F G H ")

        for i, row in enumerate(self.board):

            print(f"{i+1}|", end="") 

            for col in row:
                if col == 1:
                    print("○", end="")
                elif col == -1:
                    print("●", end="")
                else:
                    print("·", end="")
                print(" ", end="")    
            print()
            
    def get_possible_action(self, turn= None):

        if not turn:
            turn  = self.turn

        possible_acitons = []
        
        for c in range(1,9,1):
            for r in ['A','B','C','D','E','F','G','H']:

                action = f"{c}{r}"
                tmp_board = self.check_and_process_action(action)
                
                if tmp_board:
                    possible_acitons.append(action)
                    
        return possible_acitons
        
    def check_and_process_action(self, action):
        rc = pos_decode(action)
        r = rc//8
        c = rc%8

        if self.board[r][c] != 0: return

        tmp_board = deepcopy(self.board)
        reverse_pos = []

        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if x==0 and y==0: continue

                tmp_pos = []
                cur_r = r + x
                cur_c = c + y
                         
                while(cur_r >= 0 and cur_r <= 7 and cur_c >= 0 and cur_c <= 7):
                    if( tmp_board[cur_r][cur_c] == self.turn*-1 ):
                        tmp_pos.append( (cur_r, cur_c) )
                    elif( tmp_board[cur_r][cur_c] == 0 ):
                        break
                    elif ( tmp_board[cur_r][cur_c] == self.turn ):
                        reverse_pos.extend(tmp_pos)
                        break
                            
                    cur_r += x
                    cur_c += y
                
        if reverse_pos:   
            tmp_board[r][c] = self.turn
            for p in reverse_pos:
                tmp_board[p[0]][p[1]] = self.turn
            
            return tmp_board
        else:
            return None
            
    def step(self, pos):
        tmp = self.check_and_process_action(pos)
        if tmp: 
            self.board = tmp
            if self.render is True:
                self.show_board()
        else: 
            self.show_board()
            print(f"invalid pos(pos: {pos}, turn: {self.turn}\n)")

        self.switch()
        possible_action = self.get_possible_action()

        if not possible_action:
            self.switch()
            possible_action = self.get_possible_action()

            if not possible_action:
                self.isEnd = True
        return 

    def reset(self, board: list[list] = B, turn = 1):
        self.board = board
        self.turn = turn
        self.save_board = []
        self.save_turn  = []
        self.isEnd = (not self.get_possible_action(BLACK)) \
                    and (not self.get_possible_action(WHITE))
        return board

    def get_result(self):
        
        score = 0

        for r in range(8):
            for c in range(8):
                score += self.board[r][c]
        if score > 0:
            return 1
        elif score < 0:
            return -1
        else:
            return 0

    def close(self):
        del self

    def get_turn(self):
        return self.turn
                
    def clone(self):
        return Reversi(deepcopy(self.board), self.turn)

    def playout(self): 
        while(not self.isEnd()):
            actions = self.get_possible_action()
            action = random.choice(list(actions.keys()))
            self.step(action)
        
        return self.get_result()

    def recover(self, actions):
        for a in actions:
            r, c = pos_decode(a.pos)
            self.board[r][c] = EMPTY
            self.turn = a.turn

    def save(self):
        self.save_board.append(deepcopy(self.board))
        self.save_turn.append(self.turn)

    def load(self):
        self.board = self.save_board.pop()
        self.turn  = self.save_turn.pop()

    def switch(self):
        self.turn *= -1
      
if __name__ == "__main__":
    
    env = Reversi(render=True)
    # player1 = ai_player(color = 1)
    # player2 = ai_player(color = -1)

    # while True:
    #     board = env.reset()
    #     env.show_board()

    #     while(not env.isEnd):
    #         turn = env.get_turn()
    #         actions = env.get_possible_action()
    #         if turn == 1:
    #             action = player1.random_action(actions)
    #         elif turn == -1:
    #             action = player2.random_action(actions)

    #         env.step(action)
    #         pass

    #     print(f"{env.get_result()} win!")
    #     pass

