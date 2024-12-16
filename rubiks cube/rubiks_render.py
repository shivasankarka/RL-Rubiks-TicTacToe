import time
import matplotlib.pyplot as plt
import numpy as np

# color_mapping = {
#     "w": [1, 1, 1],  # White
#     "g": [0, 1, 0],  # Green
#     "y": [1, 1, 0],  # Yellow
#     "b": [0, 0, 1],  # Blue
#     "o": [1, 0.5, 0],  # Orange
#     "r": [1, 0, 0]    # Red
#     }

color_mapping = {
    0: [1, 1, 1],  # White
    1: [0, 1, 0],  # Green
    2: [1, 1, 0],  # Yellow
    3: [0, 0, 1],  # Blue
    4: [1, 0.5, 0],  # Orange
    5: [1, 0, 0]    # Red
    }

cubeT = np.array([["w", "w", "w"], ["w", "w", "w"], ["w", "w", "w"]])
cubeF = np.array([["g", "g", "g"], ["g", "g", "g"], ["g", "g", "g"]])
cubeD = np.array([["y", "y", "y"], ["y", "y", "y"], ["y", "y", "y"]])
cubeB = np.array([["b", "b", "b"], ["b", "b", "b"], ["b", "b", "b"]])
cubeL = np.array([["o", "o", "o"], ["o", "o", "o"], ["o", "o", "o"]])
cubeR = np.array([["r", "r", "r"], ["r", "r", "r"], ["r", "r", "r"]])
solved_state = [cubeT, cubeF, cubeD, cubeB, cubeL, cubeR]

fig, axs = plt.subplots(3, 4, figsize=(8, 8), facecolor='gray')
plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Add space between subplots
plt.rcParams["xtick.color"] = "white"  # Set color of x-axis ticks
plt.rcParams["ytick.color"] = "white"  # Set color of y-axis ticks
plt.rcParams["text.color"] = "white"  # Set color of text
plt.rcParams["axes.labelcolor"] = "white"  # Set color of axis labels

plt.rcParams["axes.grid"] = False  # Hide grid lines
plt.rcParams["axes.spines.left"] = False  # Hide left spine
plt.rcParams["axes.spines.right"] = False  # Hide right spine
plt.rcParams["axes.spines.top"] = False  # Hide top spine
plt.rcParams["axes.spines.bottom"] = False  # Hide bottom spine
plt.rcParams["font.family"] = "Papyrus"  # Set font family
plt.rcParams["font.size"] = 12  # Set font size
plt.rcParams["axes.titlepad"] = 20  # Set padding for title
plt.rcParams["legend.frameon"] = False  # Hide legend frame
plt.rcParams["legend.facecolor"] = "black"  # Set background color of legend
plt.rcParams["legend.edgecolor"] = "white"  # Set edge color of legend
plt.rcParams["legend.fontsize"] = 10  # Set font size of legend
plt.rcParams["axes.titlecolor"] = "white"  # Set color of title
plt.rcParams["axes.titlesize"] = 14  # Set font size of title
plt.rcParams["axes.titleweight"] = "bold"  # Set font weight of title
plt.rcParams["axes.labelsize"] = 12  # Set font size of axis labels
plt.rcParams["axes.labelweight"] = "bold"  # Set font weight of axis labels

images = []

def draw_grid(ax):
    for x in [0.5, 1.5]:
        ax.axvline(x, color='k', linestyle='-', linewidth=1)
    # Draw horizontal lines
    for y in [0.5, 1.5]:
        ax.axhline(y, color='k', linestyle='-', linewidth=1)

used_axes = [(1, 1), (2, 1), (1, 3), (0, 1), (1, 0), (1, 2)]
for i, ax in np.ndenumerate(axs):
    ax.axis('off')  # Hide all axes
    if i in used_axes:  # Positions where cube faces will be shown
        # Placeholder image, will update these later
        images.append(ax.imshow(np.zeros((3, 3, 3)), interpolation='nearest'))
    else:
        images.append(None)

axs[1, 1].set_title('Top', color='black')
axs[2, 1].set_title('Front', color='black')
axs[1, 3].set_title('Down', color='black')
axs[0, 1].set_title('Back', color='black')
axs[1, 0].set_title('Left', color='black')
axs[1, 2].set_title('Right', color='black')
plt.show(block=False)

def update_plot(cube_faces):
    for i, cube_face in enumerate(cube_faces): # T, F, D, B, L, R
        axs[used_axes[i][0], used_axes[i][1]].imshow(cube_face, interpolation='nearest')
        draw_grid(axs[used_axes[i][0], used_axes[i][1]])

    plt.draw()
    plt.pause(0.75)

def convert_to_rgb(cube):
        shape = cube.shape
        rgb_cube = np.zeros((shape[0], shape[1], 3))
        for i in range(shape[0]):
            for j in range(shape[1]):
                rgb_cube[i, j] = color_mapping[cube[i, j]]
        return rgb_cube

# scramble= ["Up", "L", "F", "F", "U", "U", "Fp", "U", "U", "L", "L", "B", "L", "L", "Bp", "L", "L", "F", "F", "D", "L", "L", "F", "F", "U", "Lp", "Dp", "Bp"]
# final_state = solved_state
# complementary_state = scramble[::-1]
# solution = []
# for move in complementary_state:
#     if move.endswith("p"):
#         solution.append(move[:-1])
#     else:
#         solution.append(move + "p")

# for move in scramble:
#     final_state = calculate_state(move, final_state)

# for i, move in enumerate(solution):
    # if i == 0:
    #     time.sleep(1)
    # final_state = calculate_state(move, final_state)
    # cube_faces_rgb = [convert_to_rgb(face) for face in final_state]
    # update_plot(cube_faces_rgb)

from rubikscube import rubiks
from MCTS import MCTS

ALLOWED_MOVES = {0: 'R', 1: 'Rp', 2: 'L', 3: 'Lp', 4: 'U', 5: 'Up', 6: 'D', 7: 'Dp', 8: 'F', 9: 'Fp', 10: 'B', 11: 'Bp'}
rub = rubiks()
args = {
    'C': 1.41,
    'num_searches':1000
}

mcts = MCTS(rub, args)
state = rub.get_solved_state()
state = rub.get_next_state(state, 0)
# state = rub.get_next_state(state, 4)
# state = rub.get_next_state(state, 1)
cube_faces_rgb = [convert_to_rgb(face) for face in state]
update_plot(cube_faces_rgb)

while True:
    mcts_probs = mcts.search(state)
    print("mcts_probs", mcts_probs)
    action = np.argmax(mcts_probs)
    print("action", ALLOWED_MOVES[action])
    state = rub.get_next_state(state, action)
    cube_faces_rgb = [convert_to_rgb(face) for face in state]
    update_plot(cube_faces_rgb)
    if rub.check_if_solved(state)[1]:
        print("Cube is solved!", state[0])
        break

plt.show()
