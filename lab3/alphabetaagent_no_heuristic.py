from connect4 import Connect4
from copy import deepcopy

class AlphaBetaAgent_nh:
    def __init__(self, my_token:str):
        self.my_token = my_token
        self.enemy_token = 'o' if my_token == 'x' else 'x'


    def min_max(self, connect4:Connect4, max_turn:bool, depth:int,alpha:float, beta:float):
        if connect4.game_over:
            if connect4.wins == self.my_token:
                return 1
            elif connect4.wins == self.enemy_token:
                return -1
            elif connect4.wins == None:
                return 0
        if depth == 0:
            return 0
        
        poss_drops = connect4.possible_drops()

        if max_turn:
            best_val = float('-inf')
            for col in poss_drops:
                new_connect4 = deepcopy(connect4)
                new_connect4.drop_token(col)
                best_val = max(best_val, self.min_max(new_connect4,False,depth-1,alpha,beta))
                alpha = max(best_val, alpha)
                del new_connect4
                if best_val >= beta:
                    break
            return best_val
        else:
            worst_val = float('inf')
            for col in poss_drops:
                new_connect4 = deepcopy(connect4)
                new_connect4.drop_token(col)
                worst_val = min(worst_val, self.min_max(new_connect4,True, depth-1,alpha,beta))
                beta = min(beta, worst_val)
                del new_connect4
                if worst_val <= alpha:
                    break
            return worst_val
        
    def decide(self, connect4:Connect4):
        best_move = None
        alpha = float('-inf')
        best_val = float('-inf')
        for col in connect4.possible_drops():
            new_connect4 = deepcopy(connect4)
            new_connect4.drop_token(col)
            val = self.min_max(new_connect4, False, 4, alpha, float('inf'))
            if val > best_val:
                best_val = val
                best_move = col
            del new_connect4
            alpha = max(alpha,best_val)
        return best_move