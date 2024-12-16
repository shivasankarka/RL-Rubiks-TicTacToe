import numpy as np

# STATE IS DEFINED AS A 3X3 MATRIX
# ACTIONS ARE DEFINED AS A NUMBER FROM 0 TO 8 (equivalent to x,y coordinates)
# PLAYERS ARE DEFINED AS 1 OR -1

class TicTacToe:
    def __init__(self):
        self.rows: int = 3
        self.columns: int = 3
        self.state: np.ndarray = np.zeros((self.rows, self.columns))

    def __repr__(self):
        return "TicTacToe"

    def get_initial_state(self):
        return self.state

    def calculate_next_state(self, action, player):
        row_coord: int = action // self.columns
        column_coord: int = action % self.columns
        self.state[row_coord, column_coord] = player

    def get_next_state(self, action, player):
        self.calculate_next_state(action, player)
        return self.state

    def get_valid_moves_coord(self):
        valid_moves = self.state.reshape(self.state.size) == 0
        valid_moves_location = np.where(valid_moves)[0]
        return valid_moves_location

    def get_valid_moves_bool(self):
        return self.state.reshape(self.state.size) == 0

    def check_win(self, action):
        row_coord = action // self.columns
        column_coord = action % self.columns
        player = self.state[row_coord, column_coord]

        return (
            np.sum(self.state[row_coord, :]) == 3 * player
            or np.sum(self.state[:, column_coord]) == 3 * player
            or np.sum(np.diag(self.state)) == 3 * player
            or np.sum(np.diag(self.state[::-1,:])) == 3 * player
        )

    def terminate_game(self, action: int) -> bool:
        if self.check_win(action):
            return True
        if len(self.get_valid_moves_coord()) == 0:
            return True
        return False

    def change_player(self, player: int):
        self.state *= player

    def encoded_board_space(self):
        return np.stack(
            (self.state == -1, self.state == 0, self.state == 1)
        ).astype(np.float32)

tctctc = TicTacToe()
print(tctctc.get_initial_state())
print(tctctc.get_next_state(4, 1))
print(tctctc.get_valid_moves_coord())
print(tctctc.state)
print(tctctc.calculate_next_state(2, 1))
print(tctctc.calculate_next_state(6, 1))
print(tctctc.state)
print(tctctc.check_win(6))
print(tctctc.terminate_game(6))
print(tctctc.get_encoded_state())
