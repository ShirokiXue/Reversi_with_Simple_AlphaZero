from MCTS import Node


D = ['A','B','C','D','E','F','G','H']

def pos_encode(input: int):
    return f"{(input//8+1)}{D[input%8]}"

def pos_decode(pos: str):
    r = int(pos[0]) - 1
    c = ord(pos[1]) - 64 - 1
    return r*8+c

def assignResult(buffer):
    for log in buffer: # [Node, Reward]
        node = log[0]
        log[1] = node.Q_value
    return buffer

def get_all_nodes(buffer, node:Node):
    buffer.append(node)
    for key in node.children:
        get_all_nodes(buffer, node.children[key])

def get_selected_nodes(buffer, node:Node):
    if node.visits > 0:
        buffer.append(node)
    if node.parent:
        get_selected_nodes(buffer, node.parent)