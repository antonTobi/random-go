'''
Script to batch generate random balanced opening positions.
(query code is copied from https://github.com/lightvector/KataGo/blob/master/python/query_analysis_engine_example.py)
'''

import argparse
import json
import subprocess
import time
from threading import Thread
import sgfmill
import sgfmill.boards
import sgfmill.ascii_boards
import sgfmill.sgf
# https://mjw.woodcraft.me.uk/sgfmill/doc/1.1.1/
from typing import Tuple, List, Optional, Union, Literal
import pprint
import os
import random

Color = Union[Literal["b"],Literal["w"]]
Move = Union[Literal["pass"],Tuple[int,int]]

def sgfmill_to_str(move: Move) -> str:
    if move == "pass":
        return "pass"
    (y,x) = move
    return "ABCDEFGHJKLMNOPQRSTUVWXYZ"[x] + str(y+1)

def str_to_sgfmill(s: str) -> Move:
    if s == "pass":
        return "pass"
    y = int(s[1:])-1
    x = "ABCDEFGHJKLMNOPQRSTUVWXYZ".index(s[0])
    return (y, x)

class KataGo:

    def __init__(self, katago_path: str, config_path: str, model_path: str):
        self.query_counter = 0
        katago = subprocess.Popen(
            [katago_path, "analysis", "-config", config_path, "-model", model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.katago = katago
        def printforever():
            while katago.poll() is None:
                data = katago.stderr.readline()
                time.sleep(0)
                if data:
                    print("KataGo: ", data.decode(), end="")
            data = katago.stderr.read()
            if data:
                print("KataGo: ", data.decode(), end="")
        self.stderrthread = Thread(target=printforever)
        # self.stderrthread.start()

    def close(self):
        self.katago.stdin.close()

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

        self.katago.stdin.write((json.dumps(query) + "\n").encode())
        self.katago.stdin.flush()

        line = ""
        while line == "":
            if self.katago.poll():
                time.sleep(1)
                raise Exception("Unexpected katago exit")
            line = self.katago.stdout.readline()
            line = line.decode().strip()
        response = json.loads(line)

        return response

def create_sgf_with_branches(size: int, komi: float, positions: List[str], max_score_lead: float, model_name: str) -> str:
    # Start with the header and add generation info comment
    sgf = f"(;GM[1]FF[4]CA[UTF-8]RU[Chinese]KM[{komi}]SZ[{size}]C[Rules: Chinese\nKomi: {komi}\nMax score lead: {max_score_lead}\nNetwork: {model_name}\nDate: {time.strftime('%Y-%m-%d')}]"
    
    # Add each position as a branch
    for i, pos in enumerate(positions, 1):
        # Convert the position string to moves
        moves = []
        for j in range(1, len(pos), 2):
            x = ord(pos[j]) - ord('a')
            y = ord(pos[j+1]) - ord('a')
            color = "B" if (j//2) % 2 == 0 else "W"
            moves.append(f"{color}[{chr(ord('a') + x)}{chr(ord('a') + y)}]")
        
        # Add the branch with a leading semicolon and position number comment
        sgf += f"(;{';'.join(moves)};C[Position #{i}])"
    
    sgf += ")"
    return sgf

def generate_positions(
    katago: KataGo,
    size: int,
    number_of_moves: int,
    number_of_boards: int,
    max_score_lead: float,
    komi: float = 7.0
) -> List[str]:
    positions = []
    prefix = "abcdefghijklmnopqrstuvwxyz"[size]
    tries = 0
    seen_positions = set()  # Track unique positions
    
    while len(positions) < number_of_boards:
        tries += 1
        board = sgfmill.boards.Board(size)
        coords = []
        
        for i in range(number_of_moves):
            # Find a random unoccupied spot
            while True:
                x = random.randint(0, size-1)
                y = random.randint(0, size-1)
                if board.get(y, x) is None:
                    break
                    
            color = "bw"[i % 2]
            board.play(y, x, color)
            coords.append("abcdefghijklmnopqrstuvwxyz"[x])
            coords.append("abcdefghijklmnopqrstuvwxyz"[y])
        
        position = prefix + "".join(coords)
        
        # Skip if we've seen this position before
        if position in seen_positions:
            continue
        seen_positions.add(position)
        
        # Progressive evaluation with increasing visit counts
        visit_counts = [4, 16, 64, 256]
        position_valid = True
        
        for visit_count in visit_counts:
            response = katago.query(board, [], komi, max_visits=visit_count)
            lead = response['rootInfo']['scoreLead']
            
            if abs(lead) >= max_score_lead:
                position_valid = False
                break
        
        if position_valid:
            positions.append(position)
            print(f"Found position {len(positions)}/{number_of_boards}")
    
    # Sort positions lexicographically
    positions.sort()
    return positions

def generate_sgf_file(
    katago: KataGo,
    size: int,
    number_of_moves: int,
    number_of_boards: int,
    model_name: str,
    max_score_lead: float,
    komi: float = 7.0
) -> str:
    positions = generate_positions(
        katago, size, number_of_moves, number_of_boards, max_score_lead, komi
    )
    
    sgf_content = create_sgf_with_branches(
        size, komi, positions, max_score_lead, model_name
    )
    
    # Generate filename based on settings
    filename = f"{size}x{size}_{number_of_moves}moves.sgf"
    
    # Save the SGF file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(sgf_content)
    
    print(f"\nSaved {len(positions)} positions to {filename}")
    return filename

if __name__ == "__main__":    
    katago_folder = "..\\..\\KataGo\\katago-v1.14.1-opencl-windows-x64"
    katago_path = os.path.join(katago_folder, "katago.exe")
    config_path = os.path.join(katago_folder, "analysis_example.cfg")
    model_path = os.path.join(katago_folder, "kata1-b28c512nbt-s8476434688-d4668249792.bin.gz")
    
    # Extract model name from the model path
    model_name = os.path.basename(model_path).split('.')[0]
    
    katago = KataGo(katago_path, config_path, model_path)

    boards_per_file = 100
    
    configurations = [
        (19, 8, boards_per_file),
        (19, 16, boards_per_file),
        (19, 24, boards_per_file),
        (13, 4, boards_per_file),
        (13, 8, boards_per_file),
        (13, 12, boards_per_file),
        (9, 2, boards_per_file),
        (9, 4, boards_per_file),
        (9, 6, boards_per_file),
    ]
    
    # Generate SGF files for each configuration
    for size, moves, boards in configurations:
        print(f"\nGenerating {size}x{size} board with {moves} moves")
        generate_sgf_file(katago, size, moves, boards, model_name, max_score_lead=0.3)
    
    katago.close() 