import numpy as np

ALLOWED_MOVES = {0: 'R', 1: 'Rp', 2: 'L', 3: 'Lp', 4: 'U', 5: 'Up', 6: 'D', 7: 'Dp', 8: 'F', 9: 'Fp', 10: 'B', 11: 'Bp'}

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children = []
        self.visit_count = 0
        self.sum_value = 0

    def is_fully_expanded(self):
        return len(self.children) == len(ALLOWED_MOVES.keys())

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            return float('inf')
        else:
            # Calculate the UCB value normally if the node has been visited
            # q_value = ((child.sum_value / child.visit_count) + 1) / 2
            q_value = child.sum_value / child.visit_count
            second_term = self.args['C'] * np.sqrt(np.log(self.visit_count) / child.visit_count)
            return q_value + second_term

    def expand(self):
        untried_actions = [action for action in ALLOWED_MOVES.keys() if action not in [child.action_taken for child in self.children]]
        action = np.random.choice(untried_actions)
        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, action)
        child = Node(self.game, self.args, child_state, self, action)
        self.children.append(child)
        return child

    def simulate(self):
        value, is_terminal = self.game.get_value_and_terminate(self.state, self.action_taken)
        if is_terminal:
            return value

        rollout_state = self.state.copy()
        # while True:
        for i in range(20):
            # print("i", i)
            # while True:
            #     action = np.random.choice(list(ALLOWED_MOVES.keys()))
            #     new_val = self.game.get_intermediate_score(rollout_state, action)
            #     if new_val > current_val:
            #         current_val = new_val
            #         print("new_val", new_val)
            #         break
            # val_moves = [self.game.get_intermediate_score(rollout_state, action) for action in list(ALLOWED_MOVES.keys())]
            # max_val_moves_index = np.argmax(val_moves)
            # action = max_val_moves_index
            val_moves = []
            for action in list(ALLOWED_MOVES.keys()):
                temp_state = self.game.get_next_state(rollout_state, action)
                val_moves.append(self.game.calculate_distance(temp_state))
            action = np.argmin(val_moves)

            # untried_actions = [action for action in ALLOWED_MOVES.keys() if action not in [child.action_taken for child in self.children]]
            # action = np.random.choice(untried_actions)
            rollout_state = self.game.get_next_state(rollout_state, action)
            value, is_terminal = self.game.check_if_solved(rollout_state)
            if is_terminal:
                return value

        return value

    def backpropagate(self, value):
        self.sum_value += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(value)

class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args

    def search(self, state):
        print("MCTS")
        root = Node(self.game, self.args, state)

        for search in range(self.args['num_searches']):
            node = root
            while node.is_fully_expanded():
                selected_node = node.select()
                node = selected_node

            value, is_terminal = self.game.get_value_and_terminate(node.state, node.action_taken)
            if not is_terminal:
                node = node.expand()
                value = node.simulate()

            node.backpropagate(value)

        # action_probs = np.zeros(self.game.action_size)
        action_probs = np.zeros(13)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs = action_probs / np.sum(action_probs)
        return action_probs

    # def search(self, state):
    #     root = Node(self.game, self.args, state)

    #     for _ in range(self.args['num_searches']):
    #         node = root
    #         # Navigate down the tree to the first non-fully expanded node
    #         while node.is_fully_expanded() and not node.game.check_if_solved(node.state):
    #             node = node.select()
    #             if node is None:
    #                 break

    #         if node is None or node.game.check_if_solved(node.state):
    #             continue

    #         if not node.is_fully_expanded():
    #             node = node.expand()

    #         value = node.simulate()
    #         node.backpropagate(value)

    #     # Calculate action probabilities based on visit counts
    #     action_probs = np.zeros(len(ALLOWED_MOVES))
    #     for child in root.children:
    #         action_index = list(ALLOWED_MOVES.keys()).index(child.action_taken)
    #         action_probs[action_index] = child.visit_count
    #     action_probs /= np.sum(action_probs)
    #     return action_probs