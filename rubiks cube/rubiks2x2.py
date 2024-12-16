# python code to calculate a state of 3*3 rubik's cube given an move input

from typing import List
import numpy as np
from sklearn import preprocessing

colors = {"w": 0, "g": 1, "y": 2, "b": 3, "o": 4, "r": 5}
# Front face of cube is always the green face and white is the top face

# solved state is defined by the colors of the faces
# cube_F = np.array([[colors["g"], colors["g"], colors["g"]], [colors["g"], colors["g"], colors["g"]], [colors["g"], colors["g"], colors["g"]]])
# cube_T = np.array([[colors["w"], colors["w"], colors["w"]], [colors["w"], colors["w"], colors["w"]], [colors["w"], colors["w"], colors["w"]]])
# cube_B = np.array([[colors["b"], colors["w"], coplors["w"]], [colors["w"], colors["w"], colors["w"]], [colors["w"], colors["w"], colors["w"]]])
# cube_D = np.array([[colors["y"], colors["y"], colors["y"]], [colors["y"], colors["y"], colors["y"]], [colors["y"], colors["y"], colors["y"]]])
# cube_L = np.array([[colors["o"], colors["o"], colors["o"]], [colors["o"], colors["o"], colors["o"]], [colors["o"], colors["o"], colors["o"]]])
# cube_R = np.array([[colors["r"], colors["r"], colors["r"]], [colors["r"], colors["r"], colors["r"]], [colors["r"], colors["r"], colors["r"]]])

class Rubiks2x2():
    def __init__(self):
        white   = np.array([1, 0, 0, 0, 0, 0])
        green   = np.array([0, 1, 0, 0, 0, 0])
        yellow  = np.array([0, 0, 1, 0, 0, 0])
        blue    = np.array([0, 0, 0, 1, 0, 0])
        orange  = np.array([0, 0, 0, 0, 1, 0])
        red     = np.array([0, 0, 0, 0, 0, 1])

        cubeT = np.array([[white, white], [white, white]])
        cubeF = np.array([[green, green], [green, green]])
        cubeD = np.array([[yellow, yellow], [yellow, yellow]])
        cubeB = np.array([[blue, blue], [blue, blue]])
        cubeL = np.array([[orange, orange], [orange, orange]])
        cubeR = np.array([[red, red], [red, red]])
        self.solved_state = np.vstack([cubeT, cubeF, cubeD, cubeB, cubeL, cubeR])

        self.state = np.vstack([cubeT, cubeF, cubeD, cubeB, cubeL, cubeR])

    def calculate(self, move: str):
        calculate_state(move, self.state)

    def reset(self):
        self.state = self.solved_state

    def scramble(self, moves: List[str]):
        for move in moves:
            self.calculate(move)

    def is_solved(self):
        return np.all(self.state == self.solved_state)

def calculate_state(move: str, state: np.ndarray):
    moves = ["R", "Rp", "L", "Lp", "U", "Up", "D", "Dp", "F", "Fp", "B", "Bp", "M", "Mp"]

    if move == "R": # * checked
        tempT = state[0:2][:,1].copy()
        tempF = state[2:4][:,1].copy()
        tempD = state[4:6][:,1].copy()
        tempB = state[6:8][:,0].copy()

        state[0:2][:,1] = tempF
        state[2:4][:,1] = tempD
        state[4:6][:,1] = np.flip(tempB)
        state[6:8][:,0] = np.flip(tempT)

        state[10:12] = np.rot90(state[10:12], -1)
        return state

    if move == "Rp": # * checked
        tempT = state[0:2][:,1].copy()
        tempF = state[2:4][:,1].copy()
        tempD = state[4:6][:,1].copy()
        tempB = state[6:8][:,0].copy()

        state[0:2][:,1] = np.flip(tempB)
        state[2:4][:,1] = tempT
        state[4:6][:,1] = tempF
        state[6:8][:,0] = np.flip(tempD)

        state[10:12] = np.rot90(state[10:12], 1)
        return state

    if move == "U": # * checked
        tempF = state[2:4][0,:].copy()
        tempB = state[6:8][0,:].copy()
        tempL = state[8:10][0,:].copy()
        tempR = state[10:12][0,:].copy()

        state[2:4][0,:] = tempR
        state[6:8][0,:] = tempL
        state[8:10][0,:] = tempF
        state[10:12][0,:] = tempB

        state[0:2] = np.rot90(state[0:2],-1)
        return state

    if move == "Up": # * checked
        tempF = state[2:4][0,:].copy()
        tempB = state[6:8][0,:].copy()
        tempL = state[8:10][0,:].copy()
        tempR = state[10:12][0,:].copy()

        state[2:4][0,:] = tempL
        state[6:8][0,:] = tempR
        state[8:10][0,:] = tempB
        state[10:12][0,:] = tempF

        state[0:2] = np.rot90(state[0:2], 1)

        return state

    if move == "L": # * checked
        tempT = state[0:2][:,0].copy()
        tempF = state[2:4][:,0].copy()
        tempD = state[4:6][:,0].copy()
        tempB = state[6:8][:,1].copy()

        state[0:2][:,0] = np.flip(tempB)
        state[2:4][:,0] = tempT
        state[4:6][:,0] = tempF
        state[6:8][:,1] = np.flip(tempD)

        state[8:10] = np.rot90(state[8:10], -1)

        return state

    if move == "Lp": # * checked
        tempT = state[0:2][:,0].copy()
        tempF = state[2:4][:,0].copy()
        tempD = state[4:6][:,0].copy()
        tempB = state[6:8][:,1].copy()

        state[0:2][:,0] = tempF
        state[2:4][:,0] = tempD
        state[4:6][:,0] = np.flip(tempB)
        state[6:8][:,1] = np.flip(tempT)

        state[8:10] = np.rot90(state[8:10], 1)

        return state

    if move == "D": # * checked
        tempF = state[2:4][1,:].copy()
        tempB = state[6:8][1,:].copy()
        tempL = state[8:10][1,:].copy()
        tempR = state[10:12][1,:].copy()

        state[2:4][1,:] = tempL
        state[6:8][1,:] = tempR
        state[8:10][1,:] = tempB
        state[10:12][1,:] = tempF

        state[4:6] = np.rot90(state[4:6], -1)

        return state

    if move == "Dp": # * checked
        tempF = state[2:4][1,:].copy()
        tempB = state[6:8][1,:].copy()
        tempL = state[8:10][1,:].copy()
        tempR = state[10:12][1,:].copy()

        state[2:4][1,:] = tempR
        state[6:8][1,:] = tempL
        state[8:10][1,:] = tempF
        state[10:12][1,:] = tempB

        state[4:6] = np.rot90(state[4:6], 1)

        return state

    if move == "F": # * checked
        tempT = state[0:2][1,:].copy()
        tempR = state[10:12][:,0].copy()
        tempD = state[4:6][0,:].copy()
        tempL = state[8:10][:,1].copy()

        state[0:2][1,:] = np.flip(tempL)
        state[10:12][:,0] = tempT
        state[4:6][0,:] = np.flip(tempR)
        state[8:10][:,1] = tempD

        state[2:4] = np.rot90(state[2:4], -1)

        return state

    if move == "Fp": # * checked
        tempT = state[0:2][1,:].copy()
        tempR = state[10:12][:,0].copy()
        tempD = state[4:6][0,:].copy()
        tempL = state[8:10][:,1].copy()

        state[0:2][1,:] = tempR
        state[10:12][:,0] = np.flip(tempD)
        state[4:6][0,:] = tempL
        state[8:10][:,1] = np.flip(tempT)

        state[2:4] = np.rot90(state[2:4], 1)
        return state

    if move == "B": # * checked
        tempT = state[0:2][0,:].copy()
        tempR = state[10:12][:,1].copy()
        tempD = state[4:6][1,:].copy()
        tempL = state[8:10][:,0].copy()

        state[0:2][0,:] = tempR
        state[10:12][:,1] = np.flip(tempD)
        state[4:6][1,:] = tempL
        state[8:10][:,0] = np.flip(tempT)

        state[6:8] = np.rot90(state[6:8], -1)

        return state

    if move == "Bp": # * checked
        tempT = state[0:2][0,:].copy()
        tempR = state[10:12][:,1].copy()
        tempD = state[4:6][1,:].copy()
        tempL = state[8:10][:,0].copy()

        state[0:2][0,:] = np.flip(tempL)
        state[10:12][:,1] = tempT
        state[4:6][1,:] = np.flip(tempR)
        state[8:10][:,0] = tempD

        state[6:8] = np.rot90(state[6:8], 1)

        return state

    if move == "M": # * checked
        tempT = state[0:2][:,1].copy()
        tempF = state[2:4][:,1].copy()
        tempD = state[4:6][:,1].copy()
        tempB = state[6:8][:,1].copy()

        state[0:2][:,1] = tempF
        state[2:4][:,1] = tempD
        state[4:6][:,1] = np.flip(tempB)
        state[6:8][:,1] = np.flip(tempT)

        return state

    if move == "Mp": # * checked
        tempT = state[0:2][:,1].copy()
        tempF = state[2:4][:,1].copy()
        tempD = state[4:6][:,1].copy()
        tempB = state[6:8][:,1].copy()

        state[0:2][:,1] = np.flip(tempB)
        state[2:4][:,1] = tempT
        state[4:6][:,1] = tempF
        state[6:8][:,1] = np.flip(tempD)

        return state

    if move not in moves:
        print("Invalid move")
        return state

if __name__ == "__main__":
    white   = np.array([1, 0, 0, 0, 0, 0])
    green   = np.array([0, 1, 0, 0, 0, 0])
    yellow  = np.array([0, 0, 1, 0, 0, 0])
    blue    = np.array([0, 0, 0, 1, 0, 0])
    orange  = np.array([0, 0, 0, 0, 1, 0])
    red     = np.array([0, 0, 0, 0, 0, 1])

    # cubeT = np.array([[colors["w"], colors["w"]], [colors["w"], colors["w"]]])
    # cubeF = np.array([[colors["g"], colors["g"]], [colors["g"], colors["g"]]])
    # cubeD = np.array([[colors["y"], colors["y"]], [colors["y"], colors["y"]]])
    # cubeB = np.array([[colors["b"], colors["b"]], [colors["b"], colors["b"]]])
    # cubeL = np.array([[colors["o"], colors["o"]], [colors["o"], colors["o"]]])
    # cubeR = np.array([[colors["r"], colors["r"]], [colors["r"], colors["r"]]])

    cubeT = np.array([[white, white], [white, white]])
    cubeF = np.array([[green, green], [green, green]])
    cubeD = np.array([[yellow, yellow], [yellow, yellow]])
    cubeB = np.array([[blue, blue], [blue, blue]])
    cubeL = np.array([[orange, orange], [orange, orange]])
    cubeR = np.array([[red, red], [red, red]])

    initial_state = [cubeT, cubeF, cubeD, cubeB, cubeL, cubeR]
    initial_state = np.vstack(initial_state)
    print("initial_state", initial_state)

    scrambled_state = ["Up", "L", "F", "F", "U", "U", "Fp", "U", "U", "L", "L", "B", "L", "L", "Bp", "L", "L", "F", "F", "D", "L", "L", "F", "F", "U", "Lp", "Dp", "Bp"]
    # scrambled_state = ["Up", "L"]

    c_map = {0:"w", 1:"g", 2:"y", 3:"b", 4:"o", 5:"r"}

    final_state = initial_state
    for scramble in scrambled_state:
        final_state = calculate_state(scramble, final_state)

    print("scrambled", final_state)

    # the solution is the inverse complementary of scrambled_state
    complementary_state = scrambled_state[::-1]
    solution = []
    for move in complementary_state:
        if move.endswith("p"):
            solution.append(move[:-1])
        else:
            solution.append(move + "p")

    print("solution: ", solution)
    for scramble in solution:
        final_state = calculate_state(scramble, final_state)

    print("final_state", final_state)