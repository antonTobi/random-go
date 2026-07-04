import argparse
import json
import random
import subprocess
import time
from threading import Thread
import sgfmill
import sgfmill.boards
import sgfmill.ascii_boards
from typing import Tuple, List, Optional, Union, Literal, Any, Dict
import os
import matplotlib.pyplot as plt
import sgfmill.sgf

Color = Union[Literal["b"],Literal["w"]]
Move = Union[None,Literal["pass"],Tuple[int,int]]

def sgfmill_to_str(move: Move) -> str:
    if move is None:
        return "pass"
    if move == "pass":
        return "pass"
    (y,x) = move
    return "ABCDEFGHJKLMNOPQRSTUVWXYZ"[x] + str(y+1)

class KataGo:

    def __init__(self, katago_path: str, config_path: str, model_path: str, additional_args: List[str] = []):
        self.query_counter = 0
        katago = subprocess.Popen(
            [katago_path, "analysis", "-config", config_path, "-model", model_path, *additional_args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.katago = katago
        # def printforever():
        #     while katago.poll() is None:
        #         data = katago.stderr.readline()
        #         time.sleep(0)
        #         if data:
        #             print("KataGo: ", data.decode(), end="")
        #     data = katago.stderr.read()
        #     if data:
        #         print("KataGo: ", data.decode(), end="")
        # self.stderrthread = Thread(target=printforever)
        # self.stderrthread.start()

    def close(self):
        if self.katago.stdin:
            self.katago.stdin.close()
        self.katago.terminate()
        self.katago.wait()


    def query(self, initial_board: sgfmill.boards.Board, moves: List[Tuple[Color,Move]], komi: float, max_visits=None):
        query = {}

        query["id"] = str(self.query_counter)
        self.query_counter += 1

        query["moves"] = [(color,sgfmill_to_str(move)) for color, move in moves]
        query["initialStones"] = []
        for y in range(initial_board.side):
            for x in range(initial_board.side):
                color = initial_board.get(y,x)
                if color:
                    query["initialStones"].append((color,sgfmill_to_str((y,x))))
        query["rules"] = "Chinese"
        query["komi"] = komi
        query["boardXSize"] = initial_board.side
        query["boardYSize"] = initial_board.side
        query["includePolicy"] = True
        if max_visits is not None:
            query["maxVisits"] = max_visits
        return self.query_raw(query)

    def query_raw(self, query: Dict[str,Any]):
        self.katago.stdin.write((json.dumps(query) + "\n").encode())
        self.katago.stdin.flush()

        # print(json.dumps(query))

        line = ""
        while line == "":
            if self.katago.poll():
                time.sleep(1)
                raise Exception("Unexpected katago exit")
            line = self.katago.stdout.readline()
            line = line.decode().strip()
            # print("Got: " + line)
        response = json.loads(line)

        # print(response)
        return response

katago_folder = "..\\..\\KataGo\\katago-v1.16.4-opencl-windows-x64"
katago_path = os.path.join(katago_folder, "katago.exe")
config_path = os.path.join(katago_folder, "analysis_example.cfg")
# model_path = os.path.join(katago_folder, "kata1-b28c512nbt-s8476434688-d4668249792.bin.gz")
model_path = os.path.join(katago_folder, "kata1-b18c384nbt-s9131461376-d4087399203.bin.gz")

model_name = os.path.basename(model_path).split('.')[0]

katago = KataGo(katago_path, config_path, model_path)

size = 19

boards_per_stone = 100
visits = 2

scoreleads_by_n: Dict[int, List[float]] = {}
all_scoreleads: List[float] = []
min_score = None
max_score = None
min_board = None
max_board = None

r = 1 # free radius around stone
d = 1 # free distance from edge

def find_empty_spot(board: sgfmill.boards.Board, r: int, d: int) -> Tuple[int, int]:
    size = board.side
    while True:
        x = random.randint(d, size-1 - d)
        y = random.randint(d, size-1 - d)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if board.get(y + dy, x + dx) is not None:
                    break
            else:
                continue
            break
        else:
            return (x, y)

def place_handicap_stones(board: sgfmill.boards.Board, n: int):
    assert(board.side == 19)
    starpoints = [(y,x) for x in [3, 9, 15] for y in [3, 9, 15]]
    indices = {
        2: [0, 8],
        3: [0, 2, 8],
        4: [0, 2, 6, 8],
        5: [0, 2, 4, 6, 8],
        6: [0, 1, 2, 6, 7, 8],
        7: [0, 1, 2, 4, 6, 7, 8],
        8: [0, 1, 2, 3, 5, 6, 7, 8],
        9: [0, 1, 2, 3, 4, 5, 6, 7, 8]
    }
    for index in indices[n]:
        (y, x) = starpoints[index]
        board.play(y, x, "b")

# Generate random boards per stone count
for n in range(2, 10):
    scoreleads_by_n[n] = []
    for board_idx in range(boards_per_stone):
        board = sgfmill.boards.Board(size)
        placed = 0
        while placed < n:
            (x, y) = find_empty_spot(board, r, d)
            board.play(y, x, "b")
            placed += 1
        if (board_idx % 10 == 0) or (board_idx == boards_per_stone - 1):
            print(f"Analyzing {n} stones, board {board_idx + 1}/{boards_per_stone}")
        results = katago.query(board, moves=[], komi=0, max_visits=visits)
        scorelead = results['rootInfo']['scoreLead']
        scoreleads_by_n[n].append(scorelead)
        all_scoreleads.append(scorelead)
        if (min_score is None) or (scorelead < min_score):
            min_score = scorelead
            min_board = board.copy()
        if (max_score is None) or (scorelead > max_score):
            max_score = scorelead
            max_board = board.copy()

# Evaluate standard handicap positions for reference
standard_handicap_scores = {}
for n in range(2, 10):
    board = sgfmill.boards.Board(19)
    place_handicap_stones(board, n)
    results = katago.query(board, moves=[], komi=0, max_visits=visits)
    scorelead = results['rootInfo']['scoreLead']
    standard_handicap_scores[n] = scorelead
    print(f"Standard {n}-stone handicap scoreLead: {scorelead:.2f}")

katago.close()

print(f"\nLowest scorelead: {min_score}")
print(sgfmill.ascii_boards.render_board(min_board))
print(f"\nHighest scorelead: {max_score}")
print(sgfmill.ascii_boards.render_board(max_board))

# Calculate statistics
import numpy as np

# Calculate overall statistics for reference
mean_all = np.mean(all_scoreleads)
std_all = np.std(all_scoreleads, ddof=1)
conf_int_all = 1.96 * std_all / np.sqrt(len(all_scoreleads))

# Define per-stone-count colors matching viridis
colors = plt.cm.viridis(np.linspace(0, 1, 8))

# Determine bins covering all data
bins = np.arange(int(np.floor(min(all_scoreleads))) - 0.5,
                 int(np.ceil(max(all_scoreleads))) + 1.5, 1)

# Plot per-stone-count histograms with matching colors
for idx, n in enumerate(range(2, 10)):
    plt.hist(scoreleads_by_n[n], bins=bins, alpha=0.5, color=colors[idx],
             edgecolor='black', linewidth=0.3,
             label=f'{n} stones (random)')

# Plot per-group means as solid lines
for idx, n in enumerate(range(2, 10)):
    mean_score = np.mean(scoreleads_by_n[n])
    plt.axvline(x=mean_score, color=colors[idx], linestyle='-', linewidth=2,
                label=f'{n}-stone mean: {mean_score:.1f}')

# Plot standard handicap scores as dashed lines
for idx, n in enumerate(range(2, 10)):
    score = standard_handicap_scores[n]
    plt.axvline(x=score, color=colors[idx], linestyle='--', linewidth=1.5,
                label=f'{n}-stone std: {score:.1f}')

plt.xlabel('ScoreLead')
plt.ylabel('Frequency')
plt.title(f'ScoreLead per stone count ({boards_per_stone} boards each, d={d}, r={r})')
plt.legend(fontsize=7, loc='upper left', ncol=2)

# Display overall statistics on the plot
# stats_text = f"Overall mean: {mean_all:.2f}\nStd dev: {std_all:.2f}\n95% CI: [{mean_all-conf_int_all:.2f}, {mean_all+conf_int_all:.2f}]"
# plt.gca().text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
#                fontsize=10, verticalalignment='top', horizontalalignment='right',
#                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Also print per-group means
print("\nPer-group mean scoreleads:")
for n in range(2, 10):
    print(f"  {n} stones: {np.mean(scoreleads_by_n[n]):.2f}")

plt.show()



