import numpy as np
import random
from MCTS import *

ALLOWED_MOVES_NC = {0: 'R', 1: 'Rp', 2: 'L', 3: 'Lp', 4: 'U', 5: 'Up', 6: 'D', 7: 'Dp', 8: 'F', 9: 'Fp', 10: 'B', 11: 'Bp'}
ALLOWED_MOVES_CN = {'R':0, 'Rp':1, 'L':2, 'Lp':3, 'U':4, 'Up':5, 'D':6, 'Dp':7, 'F':8, 'Fp':9, 'B':10, 'Bp':11}
colors = {"w": 0, "g": 1, "y": 2, "b": 3, "o": 4, "r": 5}

class rubiks():
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = len(ALLOWED_MOVES_CN.keys())

    def __repr__(self):
        return "RubiksCube"

    def get_valid_moves(self):
        pass

    def get_solved_state(self):
        cubeT = np.array([[colors["w"], colors["w"], colors["w"]], [colors["w"], colors["w"], colors["w"]], [colors["w"], colors["w"], colors["w"]]])
        cubeF = np.array([[colors["g"], colors["g"], colors["g"]], [colors["g"], colors["g"], colors["g"]], [colors["g"], colors["g"], colors["g"]]])
        cubeD = np.array([[colors["y"], colors["y"], colors["y"]], [colors["y"], colors["y"], colors["y"]], [colors["y"], colors["y"], colors["y"]]])
        cubeB = np.array([[colors["b"], colors["b"], colors["b"]], [colors["b"], colors["b"], colors["b"]], [colors["b"], colors["b"], colors["b"]]])
        cubeL = np.array([[colors["o"], colors["o"], colors["o"]], [colors["o"], colors["o"], colors["o"]], [colors["o"], colors["o"], colors["o"]]])
        cubeR = np.array([[colors["r"], colors["r"], colors["r"]], [colors["r"], colors["r"], colors["r"]], [colors["r"], colors["r"], colors["r"]]])

        return np.array([cubeT, cubeF, cubeD, cubeB, cubeL, cubeR])

    def get_scrambled_state(self):
        np.random.seed(0)
        state = self.get_solved_state()
        n = random.randint(10, 20)
        for _ in range(n):
            state = self._calculate_next_state(state, random.choice(list(ALLOWED_MOVES.keys())) )
        return state

    def get_next_state(self, state:list[np.ndarray], action: str) -> list[np.ndarray]:
        return self._calculate_next_state(state, action)

    def check_if_solved(self, state):
        count = 0
        for i in range(6):
            if np.all(state[i] == i):
                count += 1
        if count == 6:
            return 1000, True
        return -10, False

    def calculate_distance(self, state):
        copy_state = state.copy()
        val = np.array([])
        for i in range(6):
            copy_state[i] -= np.full((3,3), i)
            # val.append(np.sum(copy_state[i]))
            val = np.append(val, np.sum(copy_state[i]))
        return np.sum(val**2)

    def get_intermediate_score(self, state, action=None):
        if action is not None:
            new_state = self.get_next_state(state, action)
            # count = 0
            # for i in range(6):
            #     count += np.count_nonzero(new_state[i] == i)
            # return count
            return np.sum(self.calculate_distance(new_state))
        else:
            # count = 0
            # for i in range(6):
            #     count += np.count_nonzero(state[i] == i)
            # return count
            return np.sum(self.calculate_distance(state))

    def get_value_and_terminate(self, state, action):
        if action is None:
            return 0, False
        new_state = self.get_next_state(state, action)
        output = self.check_if_solved(new_state)
        if output[1]:
            return output[0], True
        return output[0], False

    def _calculate_next_state(self, state, move_idx):

        if move_idx is None:
            return state

        if move_idx not in ALLOWED_MOVES_NC.keys():
            print("Move doesn't exist")
            return state

        move = ALLOWED_MOVES_NC[move_idx]
        if move == "R": # * checked
            tempT = state[0][:,2].copy()
            tempF = state[1][:,2].copy()
            tempD = state[2][:,2].copy()
            tempB = state[3][:,0].copy()

            state[0][:,2] = tempF
            state[1][:,2] = tempD
            state[2][:,2] = np.flip(tempB)
            state[3][:,0] = np.flip(tempT)

            state[5] = np.rot90(state[5], -1)
            return state

        if move == "Rp": # * checked
            tempT = state[0][:,2].copy()
            tempF = state[1][:,2].copy()
            tempD = state[2][:,2].copy()
            tempB = state[3][:,0].copy()

            state[0][:,2] = np.flip(tempB)
            state[1][:,2] = tempT
            state[2][:,2] = tempF
            state[3][:,0] = np.flip(tempD)

            state[5] = np.rot90(state[5], 1)
            return state

        if move == "U": # * checked
            tempF = state[1][0,:].copy()
            tempB = state[3][0,:].copy()
            tempL = state[4][0,:].copy()
            tempR = state[5][0,:].copy()

            state[1][0,:] = tempR
            state[3][0,:] = tempL
            state[4][0,:] = tempF
            state[5][0,:] = tempB

            state[0] = np.rot90(state[0],-1)
            return state

        if move == "Up": # * checked
            tempF = state[1][0,:].copy()
            tempB = state[3][0,:].copy()
            tempL = state[4][0,:].copy()
            tempR = state[5][0,:].copy()

            state[1][0,:] = tempL
            state[3][0,:] = tempR
            state[4][0,:] = tempB
            state[5][0,:] = tempF

            state[0] = np.rot90(state[0], 1)

            return state

        if move == "L": # * checked
            tempT = state[0][:,0].copy()
            tempF = state[1][:,0].copy()
            tempD = state[2][:,0].copy()
            tempB = state[3][:,2].copy()

            state[0][:,0] = np.flip(tempB)
            state[1][:,0] = tempT
            state[2][:,0] = tempF
            state[3][:,2] = np.flip(tempD)

            state[4] = np.rot90(state[4], -1)

            return state

        if move == "Lp": # * checked
            tempT = state[0][:,0].copy()
            tempF = state[1][:,0].copy()
            tempD = state[2][:,0].copy()
            tempB = state[3][:,2].copy()

            state[0][:,0] = tempF
            state[1][:,0] = tempD
            state[2][:,0] = np.flip(tempB)
            state[3][:,2] = np.flip(tempT)

            state[4] = np.rot90(state[4], 1)

            return state

        if move == "D": # * checked
            tempF = state[1][2,:].copy()
            tempB = state[3][2,:].copy()
            tempL = state[4][2,:].copy()
            tempR = state[5][2,:].copy()

            state[1][2,:] = tempL
            state[3][2,:] = tempR
            state[4][2,:] = tempB
            state[5][2,:] = tempF

            state[2] = np.rot90(state[2], -1)

            return state

        if move == "Dp": # * checked
            tempF = state[1][2,:].copy()
            tempB = state[3][2,:].copy()
            tempL = state[4][2,:].copy()
            tempR = state[5][2,:].copy()

            state[1][2,:] = tempR
            state[3][2,:] = tempL
            state[4][2,:] = tempF
            state[5][2,:] = tempB

            state[2] = np.rot90(state[2], 1)

            return state

        if move == "F": # * checked
            tempT = state[0][2,:].copy()
            tempR = state[5][:,0].copy()
            tempD = state[2][0,:].copy()
            tempL = state[4][:,2].copy()

            state[0][2,:] = np.flip(tempL)
            state[5][:,0] = tempT
            state[2][0,:] = np.flip(tempR)
            state[4][:,2] = tempD

            state[1] = np.rot90(state[1], -1)

            return state

        if move == "Fp": # * checked
            tempT = state[0][2,:].copy()
            tempR = state[5][:,0].copy()
            tempD = state[2][0,:].copy()
            tempL = state[4][:,2].copy()

            state[0][2,:] = tempR
            state[5][:,0] = np.flip(tempD)
            state[2][0,:] = tempL
            state[4][:,2] = np.flip(tempT)

            state[1] = np.rot90(state[1], 1)
            return state

        if move == "B": # * checked
            tempT = state[0][0,:].copy()
            tempR = state[5][:,2].copy()
            tempD = state[2][2,:].copy()
            tempL = state[4][:,0].copy()

            state[0][0,:] = tempR
            state[5][:,2] = np.flip(tempD)
            state[2][2,:] = tempL
            state[4][:,0] = np.flip(tempT)

            state[3] = np.rot90(state[3], -1)

            return state

        if move == "Bp": # * checked
            tempT = state[0][0,:].copy()
            tempR = state[5][:,2].copy()
            tempD = state[2][2,:].copy()
            tempL = state[4][:,0].copy()

            state[0][0,:] = np.flip(tempL)
            state[5][:,2] = tempT
            state[2][2,:] = np.flip(tempR)
            state[4][:,0] = tempD

            state[3] = np.rot90(state[3], 1)

            return state

        if move == "M": # * checked
            tempT = state[0][:,1].copy()
            tempF = state[1][:,1].copy()
            tempD = state[2][:,1].copy()
            tempB = state[3][:,1].copy()

            state[0][:,1] = tempF
            state[1][:,1] = tempD
            state[2][:,1] = np.flip(tempB)
            state[3][:,1] = np.flip(tempT)

            return state

        if move == "Mp": # * checked
            tempT = state[0][:,1].copy()
            tempF = state[1][:,1].copy()
            tempD = state[2][:,1].copy()
            tempB = state[3][:,1].copy()

            state[0][:,1] = np.flip(tempB)
            state[1][:,1] = tempT
            state[2][:,1] = tempF
            state[3][:,1] = np.flip(tempD)

            return state


rub = rubiks()
args = {
    'C': 1.41,
    'num_searches':3000
}

mcts = MCTS(rub, args)
state = rub.get_solved_state()
state = rub.get_next_state(state, ALLOWED_MOVES_CN['R'])
state = rub.get_next_state(state, ALLOWED_MOVES_CN['U'])
# state = rub.get_next_state(state, ALLOWED_MOVES_CN['Rp'])

# print("value", rub.calculate_distance(state))
# for move in ['R', 'F','F']:
#     state = rub.get_next_state(state, ALLOWED_MOVES_CN[move])
#     print("value", rub.calculate_distance(state))

while True:
    mcts_probs = mcts.search(state)
    print("mcts_probs", mcts_probs)
    action = np.argmax(mcts_probs)
    print("action", ALLOWED_MOVES_NC[action])
    print(action)
    if action not in ALLOWED_MOVES_NC.keys():
        print("Invalid move")
        continue

    state = rub.get_next_state(state, action)
    out = rub.check_if_solved(state)
    if out[1]:
        print("Cube is solved!", state[0])
        break


