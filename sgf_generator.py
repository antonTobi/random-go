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
from typing import Tuple, List, Optional, Union, Literal, Dict, Any
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

def create_sgf_with_branches(size: int, komi: float, positions: List[Dict[str, Any]], max_score_lead: float, model_name: str) -> str:
    """
    Create an SGF file with multiple branches, one for each position.
    Each position is set up using AB/AW properties rather than sequences of moves.
    
    Args:
        size: Board size
        komi: Komi value
        positions: List of position data dictionaries
        max_score_lead: Maximum score lead used for filtering
        model_name: Name of the KataGo model used
        
    Returns:
        SGF content as string
    """
    # Start with the header and add generation info comment
    sgf = f"(;GM[1]FF[4]CA[UTF-8]RU[Chinese]KM[{komi}]SZ[{size}]C[Rules: Chinese\nKomi: {komi}\nMax score lead: {max_score_lead}\nNetwork: {model_name}\nDate: {time.strftime('%Y-%m-%d')}]"
    
    # Add each position as a branch
    for i, position in enumerate(positions, 1):
        # Start a new branch
        sgf += "("
        
        # Add the root node with basic properties
        sgf += f";GM[1]FF[4]CA[UTF-8]RU[Chinese]KM[{komi}]SZ[{size}]"
        
        # Get lists of black and white stone coordinates
        black_stones = []
        white_stones = []
        
        # Parse the board representation
        if "board" in position:
            board_str = position["board"].replace("/", "")
            for y in range(size):
                for x in range(size):
                    index = y * size + x
                    if index < len(board_str):
                        # Convert to SGF coordinates (letters, not numbers)
                        sgf_coord = f"{chr(97 + x)}{chr(97 + y)}"
                        if board_str[index] == 'X':
                            black_stones.append(sgf_coord)
                        elif board_str[index] == 'O':
                            white_stones.append(sgf_coord)
        
        # Add black stones with AB property
        if black_stones:
            sgf += "AB"
            for coord in black_stones:
                sgf += f"[{coord}]"
        
        # Add white stones with AW property
        if white_stones:
            sgf += "AW"
            for coord in white_stones:
                sgf += f"[{coord}]"
        
        # Set player to move
        if "nextPla" in position:
            sgf += f"PL[{position['nextPla']}]"
        
        # Add a comment with the position number
        sgf += f"C[Position #{i}]"
        
        # Close the branch
        sgf += ")"
    
    # Close the main SGF
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
    """
    Generate random balanced positions and save them to an SGF file.
    
    Args:
        katago: KataGo instance
        size: Board size
        number_of_moves: Number of moves to play for each position
        number_of_boards: Number of positions to generate
        model_name: Name of the KataGo model
        max_score_lead: Maximum score lead to consider a position balanced
        komi: Komi value
        
    Returns:
        The filename of the created SGF file
    """
    # Generate random positions
    compact_positions = generate_positions(
        katago, size, number_of_moves, number_of_boards, max_score_lead, komi
    )
    
    # Convert compact positions to position dictionaries
    positions = []
    for pos in compact_positions:
        # Extract stone placements from the compact format
        board_str = ["." * size for _ in range(size)]
        for j in range(1, len(pos), 2):
            x = ord(pos[j]) - ord('a')
            y = ord(pos[j+1]) - ord('a')
            color = "X" if (j//2) % 2 == 0 else "O"
            row = list(board_str[y])
            row[x] = color
            board_str[y] = "".join(row)
        
        # Create a position dictionary
        position = {
            "board": "/".join(board_str),
            "nextPla": "B",  # Always set black to play for generated positions
            "xSize": size,
            "ySize": size,
            "komi": komi
        }
        positions.append(position)
    
    # Generate SGF content with the positions
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

def parse_startposes(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a startposes file containing Go positions in JSON format.
    
    Args:
        file_path: Path to the JSON file containing positions
        
    Returns:
        List of position data dictionaries
    """
    print(f"Parsing startposes from {file_path}")
    positions = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read file line by line as each line should be a valid JSON
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                position = json.loads(line)
                positions.append(position)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {line_num} as JSON: {line[:50]}...")
    
    print(f"Successfully parsed {len(positions)} positions")
    return positions

def convert_board_string_to_stones(board_str: str, x_size: int, y_size: int) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Convert a board string representation to a list of stones.
    
    Args:
        board_str: String representation of the board ('.' for empty, 'X' for black, 'O' for white)
        x_size: Width of the board
        y_size: Height of the board
        
    Returns:
        List of (color, position) tuples where color is 'b' or 'w' and position is (y, x)
    """
    stones = []
    
    # Remove any newlines and make sure the string has the right length
    board_str = board_str.replace('/', '')
    if len(board_str) != x_size * y_size:
        print(f"Warning: Board string has wrong length ({len(board_str)}, expected {x_size * y_size})")
        return stones
    
    for y in range(y_size):
        for x in range(x_size):
            index = y * x_size + x
            if index >= len(board_str):
                break
                
            if board_str[index] == 'X':
                stones.append(('b', (y, x)))
            elif board_str[index] == 'O':
                stones.append(('w', (y, x)))
    
    return stones

def query_positions_from_file(
    katago: KataGo,
    file_path: str,
    max_positions: int = None,
    max_visits: int = 100,
    komi: float = 7.0
) -> List[Dict[str, Any]]:
    """
    Load positions from a startposes file and query KataGo with them.
    
    Args:
        katago: KataGo instance
        file_path: Path to the file containing positions
        max_positions: Maximum number of positions to query (None for all)
        max_visits: Number of visits for KataGo analysis
        komi: Komi value to use
        
    Returns:
        List of KataGo responses
    """
    positions = parse_startposes(file_path)
    
    if max_positions is not None:
        positions = positions[:max_positions]
        
    print(f"Querying KataGo with {len(positions)} positions")
    
    results = []
    for i, position in enumerate(positions):
        # Get board size from position data
        x_size = position.get("xSize", 19)
        y_size = position.get("ySize", 19)
        board = sgfmill.boards.Board(x_size)
        
        # Parse board string if it exists
        if "board" in position:
            initial_stones = convert_board_string_to_stones(position["board"], x_size, y_size)
            for color, (y, x) in initial_stones:
                board.play(y, x, color)
        
        # Check if position has initialStones field (alternative format)
        elif "initialStones" in position:
            for color, coord in position["initialStones"]:
                move = str_to_sgfmill(coord)
                if move != "pass":
                    y, x = move
                    board.play(y, x, color)
        
        # Get next player if specified
        next_player = position.get("nextPla", "B")
        # Convert to lowercase for sgfmill format
        next_player = next_player.lower()
        
        # Get moves if they exist
        moves = position.get("moves", [])
        sgfmill_moves = []
        for color, coord in moves:
            move = str_to_sgfmill(coord)
            sgfmill_moves.append((color, move))
        
        # Use position-specific komi if provided
        position_komi = position.get("komi", komi)
        
        print(f"Querying position {i+1}/{len(positions)}")
        response = katago.query(board, sgfmill_moves, position_komi, max_visits=max_visits)
        
        # Add the original position data to the response for reference
        response["originalPosition"] = position
        results.append(response)
        
    return results

def create_sgf_from_analysis(position: Dict[str, Any], response: Dict[str, Any], filename: str = None) -> str:
    """
    Create an SGF file from a position and KataGo analysis response.
    
    Args:
        position: Position data dictionary
        response: KataGo analysis response
        filename: Optional filename to save SGF to
        
    Returns:
        SGF content as string
    """
    x_size = position.get("xSize", 19)
    y_size = position.get("ySize", 19)
    komi = position.get("komi", 7.0)
    
    # Create new SGF game
    sgf_game = sgfmill.sgf.Sgf_game(size=x_size)
    
    # Set the date correctly - use root.set directly
    root = sgf_game.get_root()
    root.set("DT", time.strftime("%Y-%m-%d"))
    
    # Set rules and komi
    root.set("KM", komi)
    root.set("RU", "Chinese")
    
    # Add KataGo analysis as comment
    if "rootInfo" in response:
        root_info = response["rootInfo"]
        comment = f"KataGo analysis:\n"
        comment += f"Score lead: {root_info.get('scoreLead', 'N/A')}\n"
        comment += f"Win rate: {root_info.get('winrate', 'N/A')}\n"
        comment += f"Visit count: {root_info.get('visits', 'N/A')}\n"
        root.set("C", comment)
    
    # Add initial stones if board string is available
    if "board" in position:
        stones = convert_board_string_to_stones(position["board"], x_size, y_size)
        black_stones = []
        white_stones = []
        
        for color, (y, x) in stones:
            if color == 'b':
                black_stones.append((x, y))
            else:
                white_stones.append((x, y))
        
        if black_stones or white_stones:
            root.set_setup_stones(black=black_stones, white=white_stones)
    
    # Add next player info if available
    if "nextPla" in position:
        next_player = position["nextPla"].lower()
        if next_player == 'b':
            root.set("PL", "b")
        else:
            root.set("PL", "w")
    
    # Add PV (Principal Variation) as a sequence of moves
    if "moveInfos" in response:
        node = root
        for move_info in response["moveInfos"]:
            if "move" in move_info and move_info["move"] != "pass":
                # Convert move to SGF coordinates (like "dd" for 3-3 point)
                move_str = move_info["move"]
                x = "ABCDEFGHJKLMNOPQRSTUVWXYZ".index(move_str[0])
                y = int(move_str[1:]) - 1
                
                # Determine whose move it is
                color = "b" if node == root and position.get("nextPla", "B").lower() == "b" else "w"
                if node != root:
                    # Alternate colors for subsequent moves
                    prev_color = node.get_move()[0]
                    color = "w" if prev_color == "b" else "b"
                
                # Add move and analysis info
                node = node.new_child()
                node.set_move(color, (x, y))
                
                # Add analysis info as comment
                comment = f"Win rate: {move_info.get('winrate', 'N/A')}\n"
                comment += f"Score lead: {move_info.get('scoreLead', 'N/A')}\n"
                comment += f"Visit count: {move_info.get('visits', 'N/A')}\n"
                node.set("C", comment)
    
    # Save to file if filename is provided
    if filename:
        with open(filename, "wb") as f:
            f.write(sgf_game.serialise())
    
    return sgf_game.serialise().decode("utf-8")

def save_analysis_as_sgf(results: List[Dict[str, Any]], prefix: str = "analysis_") -> List[str]:
    """
    Save analysis results as SGF files.
    
    Args:
        results: List of KataGo analysis responses with originalPosition
        prefix: Prefix for SGF filenames
        
    Returns:
        List of created filenames
    """
    filenames = []
    
    for i, result in enumerate(results):
        if "originalPosition" not in result:
            print(f"Warning: No original position data in result {i}, skipping")
            continue
            
        position = result["originalPosition"]
        
        # Create filename based on position properties
        x_size = position.get("xSize", 19)
        y_size = position.get("ySize", 19)
        filename = f"{prefix}{i+1}_{x_size}x{y_size}.sgf"
        
        # Create SGF file
        create_sgf_from_analysis(position, result, filename)
        filenames.append(filename)
        print(f"Saved analysis to {filename}")
    
    return filenames

def filter_balanced_positions(
    katago: KataGo,
    positions: List[Dict[str, Any]],
    max_score_lead: float,
    komi: float = 7.0,
    max_positions: int = None,
    black_to_play_only: bool = True,
    equal_stones: bool = False
) -> List[Dict[str, Any]]:
    """
    Filter positions from startposes to find balanced ones (close to 0 score lead).
    
    Args:
        katago: KataGo instance
        positions: List of positions from startposes
        max_score_lead: Maximum absolute score lead to consider a position balanced
        komi: Default komi value
        max_positions: Maximum number of balanced positions to return
        black_to_play_only: Only include positions where black is the next player
        equal_stones: Only include positions where black and white have equal number of stones
        
    Returns:
        List of balanced position dictionaries
    """
    balanced_positions = []
    tries = 0
    
    print(f"Filtering positions to find balanced ones (max score lead: {max_score_lead})")
    if black_to_play_only:
        print("Only including positions where black is the next player")
    if equal_stones:
        print("Only including positions where black and white have equal number of stones")
    
    for i, position in enumerate(positions):
        tries += 1
        
        # Check if black is the next player (if filtering is enabled)
        if black_to_play_only and position.get("nextPla", "B") != "B":
            # print(f"Position {i+1} skipped: next player is not black")
            continue
        
        # Get board size from position data
        x_size = position.get("xSize", 19)
        y_size = position.get("ySize", 19)
        
        # Check for equal number of stones if filtering is enabled
        if equal_stones and "board" in position:
            board_str = position["board"].replace("/", "")
            black_count = board_str.count("X")
            white_count = board_str.count("O")
            
            if black_count != white_count:
                # print(f"Position {i+1} skipped: unequal stones (B: {black_count}, W: {white_count})")
                continue
                
        board = sgfmill.boards.Board(x_size)
        
        # Parse board string if it exists
        if "board" in position:
            initial_stones = convert_board_string_to_stones(position["board"], x_size, y_size)
            for color, (y, x) in initial_stones:
                board.play(y, x, color)
        
        # Check if position has initialStones field (alternative format)
        elif "initialStones" in position:
            black_count = 0
            white_count = 0
            for color, coord in position["initialStones"]:
                if color == "b":
                    black_count += 1
                else:
                    white_count += 1
                    
                move = str_to_sgfmill(coord)
                if move != "pass":
                    y, x = move
                    board.play(y, x, color)
            
            # Check for equal stones in initialStones format
            if equal_stones and black_count != white_count:
                # print(f"Position {i+1} skipped: unequal stones (B: {black_count}, W: {white_count})")
                continue
        
        # Get moves if they exist
        moves = position.get("moves", [])
        sgfmill_moves = []
        for color, coord in moves:
            move = str_to_sgfmill(coord)
            sgfmill_moves.append((color, move))
        
        # Use position-specific komi if provided
        position_komi = position.get("komi", komi)
        
        # Progressive evaluation with increasing visit counts
        visit_counts = [16, 64, 256]
        position_valid = True
        
        for visit_count in visit_counts:
            print(f"Evaluating position {i+1}/{len(positions)} with {visit_count} visits")
            response = katago.query(board, sgfmill_moves, position_komi, max_visits=visit_count)
            lead = response['rootInfo']['scoreLead']
            
            if abs(lead) >= max_score_lead:
                position_valid = False
                print(f"Position {i+1} rejected: score lead {lead:.2f} exceeds threshold {max_score_lead}")
                break
        
        if position_valid:
            # Just keep the original position dictionary
            balanced_positions.append(position)
            print(f"Found balanced position {len(balanced_positions)}")
            
            if max_positions and len(balanced_positions) >= max_positions:
                break
    
    print(f"Found {len(balanced_positions)} balanced positions out of {tries} tried")
    return balanced_positions

def generate_sgf_from_startposes(
    katago: KataGo,
    positions: List[Dict[str, Any]],
    max_positions: int,
    model_name: str,
    max_score_lead: float,
    komi: float = 7.0,
    output_file: str = None,
    black_to_play_only: bool = True,
    equal_stones: bool = False
) -> str:
    """
    Generate an SGF file with multiple branches from filtered balanced positions in startposes.
    
    Args:
        katago: KataGo instance
        positions: List of positions from startposes
        max_positions: Maximum number of balanced positions to include
        model_name: Name of the KataGo model used
        max_score_lead: Maximum score lead to consider a position balanced
        komi: Default komi value
        output_file: Optional output filename (default: based on position count and board size)
        black_to_play_only: Only include positions where black is the next player
        equal_stones: Only include positions where black and white have equal number of stones
        
    Returns:
        Filename of the created SGF file
    """
    # Filter positions to find balanced ones
    balanced_positions = filter_balanced_positions(
        katago, positions, max_score_lead, komi, max_positions, black_to_play_only, equal_stones
    )
    
    if not balanced_positions:
        print("No balanced positions found.")
        return None
    
    # Get the board size from the first position
    first_pos = positions[0]
    size = first_pos.get("xSize", 19)
    
    # Create SGF content with branches
    sgf_content = create_sgf_with_branches(
        size, komi, balanced_positions, max_score_lead, model_name
    )
    
    # Generate filename based on settings if not provided
    if not output_file:
        black_to_play_str = "_black_to_play" if black_to_play_only else ""
        equal_stones_str = "_equal_stones" if equal_stones else ""
        output_file = f"balanced_startposes_{size}x{size}{black_to_play_str}{equal_stones_str}_{len(balanced_positions)}pos.sgf"
    
    # Save the SGF file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(sgf_content)
    
    print(f"\nSaved {len(balanced_positions)} balanced positions to {output_file}")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random balanced Go positions or query KataGo with existing positions")
    parser.add_argument("--mode", choices=["generate", "query", "batch"], default="generate",
                       help="Mode: generate random positions, query from JSON file, or batch create SGF from startposes")
    parser.add_argument("--startposes", type=str, help="Path to startposes JSON file for query/batch modes")
    parser.add_argument("--max-positions", type=int, default=100, help="Maximum number of positions to query/include")
    parser.add_argument("--max-visits", type=int, default=100, help="Number of visits for KataGo analysis")
    parser.add_argument("--output", type=str, help="Output file for query results (JSON format) or SGF file for batch mode")
    parser.add_argument("--komi", type=float, default=7.0, help="Default komi value to use (can be overridden by position-specific komi)")
    parser.add_argument("--katago-path", type=str, help="Path to KataGo executable")
    parser.add_argument("--config-path", type=str, help="Path to KataGo config file")
    parser.add_argument("--model-path", type=str, help="Path to KataGo model file")
    parser.add_argument("--sgf-output", action="store_true", help="Save analysis results as SGF files")
    parser.add_argument("--sgf-prefix", type=str, default="analysis_", help="Prefix for SGF filenames")
    parser.add_argument("--sgf-dir", type=str, default="analysis_sgfs", help="Directory to save SGF files")
    parser.add_argument("--max-score-lead", type=float, default=0.3, help="Maximum score lead to consider a position balanced")
    parser.add_argument("--all-players", action="store_true", help="Include positions for both black and white to play (default: black only)")
    parser.add_argument("--equal-stones", action="store_true", help="Only include positions where black and white have equal number of stones")
    args = parser.parse_args()
    
    katago_folder = "..\\..\\KataGo\\katago-v1.14.1-opencl-windows-x64"
    katago_path = args.katago_path if args.katago_path else os.path.join(katago_folder, "katago.exe")
    config_path = args.config_path if args.config_path else os.path.join(katago_folder, "analysis_example.cfg")
    model_path = args.model_path if args.model_path else os.path.join(katago_folder, "kata1-b28c512nbt-s8476434688-d4668249792.bin.gz")
    
    # Extract model name from the model path
    model_name = os.path.basename(model_path).split('.')[0]
    
    katago = KataGo(katago_path, config_path, model_path)

    if args.mode == "query" and args.startposes:
        # Create SGF output directory if needed
        if args.sgf_output and args.sgf_dir:
            os.makedirs(args.sgf_dir, exist_ok=True)
            
        # Query mode: Load positions from JSON file and query KataGo
        results = query_positions_from_file(
            katago, 
            args.startposes, 
            max_positions=args.max_positions, 
            max_visits=args.max_visits,
            komi=args.komi
        )
        
        # Save results if output file is specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Saved {len(results)} results to {args.output}")
        else:
            print("No output file specified. Here's a summary of the results:")
            for i, result in enumerate(results):
                print(f"Position {i+1}:")
                if "rootInfo" in result:
                    print(f"  Score lead: {result['rootInfo'].get('scoreLead', 'N/A')}")
                    print(f"  Win rate: {result['rootInfo'].get('winrate', 'N/A')}")
                print()
                
        # Save analysis as SGF files if requested
        if args.sgf_output:
            sgf_prefix = os.path.join(args.sgf_dir, args.sgf_prefix)
            save_analysis_as_sgf(results, prefix=sgf_prefix)
    
    elif args.mode == "batch" and args.startposes:
        # Batch mode: Create a single SGF file with balanced positions from startposes
        print(f"Batch mode: Creating SGF from balanced positions in {args.startposes}")
        
        # Parse startposes file
        positions = parse_startposes(args.startposes)
        
        # Generate SGF file with balanced positions
        generate_sgf_from_startposes(
            katago,
            positions,
            max_positions=args.max_positions,
            model_name=model_name,
            max_score_lead=args.max_score_lead,
            komi=args.komi,
            output_file=args.output,
            black_to_play_only=not args.all_players,
            equal_stones=args.equal_stones
        )
        
    else:
        # Generate mode: Create random balanced positions
        boards_per_file = args.max_positions if args.max_positions else 100
        
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
            generate_sgf_file(katago, size, moves, boards, model_name, max_score_lead=args.max_score_lead)
    
    katago.close() 