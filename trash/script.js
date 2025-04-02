// Global variables
let currentSgf = null;
let currentBranch = null;
let canvas = document.getElementById('goBoard');
let ctx = canvas.getContext('2d');
let boardSize = 19; // Default size, will be updated based on SGF

// List of available SGF files
const sgfFiles = [
    { name: '19x19 Positions (20 moves)', path: 'sgf/random_positions_19x19_20moves.sgf' },
    { name: '19x19 Positions', path: 'sgf/random_positions_19x19.sgf' },
    { name: '9x9 Positions', path: 'sgf/random_positions_9x9.sgf' }
];

// SGF parsing function (completely rewritten)
function parseSgf(sgfString) {
    console.log('Parsing SGF:', sgfString.substring(0, 100) + '...');
    
    // Create our root SGF object
    const sgf = {
        properties: {},
        branches: []
    };
    
    // Find the root node properties - match from start to first branch
    let firstBranchIndex = sgfString.indexOf('(;B[');
    if (firstBranchIndex === -1) {
        firstBranchIndex = sgfString.indexOf('(;W[');
    }
    
    if (firstBranchIndex > 0) {
        const rootNodeContent = sgfString.substring(0, firstBranchIndex);
        console.log('Root node content:', rootNodeContent);
        
        // Extract properties using regex
        const propRegex = /([A-Z]+)\[(.*?)(?:\](?!\[))/g;
        let propMatch;
        
        while ((propMatch = propRegex.exec(rootNodeContent)) !== null) {
            const key = propMatch[1];
            const value = propMatch[2];
            sgf.properties[key] = value;
            if (key === 'SZ') {
                console.log('Found board size:', value);
            }
        }
    }
    
    // Find all branches - look for specific branch start patterns
    const branches = [];
    const branchStarts = [];
    
    // Find all starting positions of branches
    const branchStartPattern = /\(;B\[|\(;W\[/g;
    let match;
    
    while ((match = branchStartPattern.exec(sgfString)) !== null) {
        branchStarts.push(match.index);
        console.log('Found branch start at', match.index);
    }
    
    // Process each branch by finding its closing parenthesis
    for (let i = 0; i < branchStarts.length; i++) {
        const startIndex = branchStarts[i];
        console.log('Processing branch at', startIndex);
        
        // Create branch object
        const branch = {
            moves: []
        };
        
        // Extract the branch content by finding the matching closing parenthesis
        let depth = 1;
        let endIndex = startIndex + 1;
        
        while (depth > 0 && endIndex < sgfString.length) {
            if (sgfString[endIndex] === '(') depth++;
            if (sgfString[endIndex] === ')') depth--;
            endIndex++;
        }
        
        const branchContent = sgfString.substring(startIndex, endIndex);
        console.log('Branch content length:', branchContent.length);
        
        // Extract moves from branch content
        const moveRegex = /;([BW])\[(..)\]/g;
        let moveMatch;
        
        while ((moveMatch = moveRegex.exec(branchContent)) !== null) {
            const color = moveMatch[1];
            const coord = moveMatch[2];
            branch.moves.push({
                color: color === 'B' ? 'black' : 'white',
                x: coord.charCodeAt(0) - 97,
                y: coord.charCodeAt(1) - 97
            });
        }
        
        if (branch.moves.length > 0) {
            branches.push(branch);
            console.log('Added branch with', branch.moves.length, 'moves');
        }
    }
    
    sgf.branches = branches;
    
    console.log('Parsed SGF with', sgf.branches.length, 'branches');
    console.log('Root properties:', sgf.properties);
    return sgf;
}

function generateNewPosition() {
    if (!currentSgf) {
        console.error('No SGF loaded');
        return;
    }
    
    // Update board size based on SGF properties
    if (currentSgf.properties.SZ) {
        const sizeStr = currentSgf.properties.SZ;
        // Handle both "9" and "9:9" formats
        boardSize = parseInt(sizeStr.split(':')[0]);
        console.log('Updated board size to:', boardSize);
    }
    
    // Get all branches
    const branches = currentSgf.branches;
    if (!branches || branches.length === 0) {
        console.error('No branches found in SGF');
        return;
    }
    
    console.log('Available branches:', branches.length);
    
    // Select a random branch
    const randomIndex = Math.floor(Math.random() * branches.length);
    currentBranch = branches[randomIndex];
    console.log('Selected branch index:', randomIndex, 'out of', branches.length);
    
    // Get stones from branch
    const stones = currentBranch.moves;
    console.log('Generated stones:', stones);
    
    // Redraw the board with the new size
    drawBoard();
    drawStones(stones);
}

// Function to resize canvas
function resizeCanvas() {
    // Get the device pixel ratio
    const dpr = window.devicePixelRatio || 1;
    
    // Get the container dimensions
    const container = document.querySelector('.board-container');
    const containerWidth = container.clientWidth;
    const containerHeight = window.innerHeight * 0.7;
    
    // Determine the size to make the canvas square
    const availableSize = Math.min(containerWidth, containerHeight);
    
    // Calculate the cell size to fit in the container
    // We want padding = cellSize, so the total width needs to be cellSize * (boardSize + 1)
    const cellSize = Math.floor(availableSize / (boardSize + 1));
    
    // Calculate the total canvas size needed (padding on both sides + grid)
    const canvasSize = cellSize * (boardSize + 1);
    
    // Set CSS size (display size)
    canvas.style.width = `${canvasSize}px`;
    canvas.style.height = `${canvasSize}px`;
    
    // Set actual canvas size accounting for device pixel ratio for crisp rendering
    canvas.width = canvasSize * dpr;
    canvas.height = canvasSize * dpr;
    
    // Scale the context to account for the device pixel ratio
    ctx.scale(dpr, dpr);
    
    // Redraw if there's a current position
    if (currentBranch) {
        drawBoard();
        drawStones(currentBranch.moves);
    }
}

// Board drawing functions
function drawBoard() {
    console.log('Drawing board with size:', boardSize);
    
    // Get the canvas size (it's a square now, so width = height)
    const canvasSize = canvas.width;
    
    // Calculate cell size (padding = cell size)
    const cellSize = canvasSize / (boardSize + 1);
    
    // Padding is equal to cellSize
    const padding = cellSize;
    
    // Calculate the actual grid size
    const gridSize = cellSize * (boardSize - 1);
    
    // Clear canvas
    ctx.fillStyle = '#DEB887';
    ctx.fillRect(0, 0, canvasSize, canvasSize);
    
    // Draw grid lines
    ctx.beginPath();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    
    for (let i = 0; i < boardSize; i++) {
        // Vertical lines - add 0.5 for pixel alignment
        const x = Math.floor(padding + i * cellSize) + 0.5;
        ctx.moveTo(x, padding + 0.5);
        ctx.lineTo(x, padding + gridSize + 0.5);
        
        // Horizontal lines - add 0.5 for pixel alignment
        const y = Math.floor(padding + i * cellSize) + 0.5;
        ctx.moveTo(padding + 0.5, y);
        ctx.lineTo(padding + gridSize + 0.5, y);
    }
    
    ctx.stroke();
    
    // Draw star points based on board size
    ctx.fillStyle = '#000';
    
    let starPoints;
    if (boardSize === 19) {
        starPoints = [3, 9, 15];
    } else if (boardSize === 13) {
        starPoints = [3, 6, 9];
    } else if (boardSize === 9) {
        starPoints = [2, 4, 6];
    } else {
        // For other sizes, use a simple pattern
        starPoints = [Math.floor(boardSize/4), Math.floor(boardSize/2), Math.floor(3*boardSize/4)];
    }
    
    for (let i = 0; i < starPoints.length; i++) {
        for (let j = 0; j < starPoints.length; j++) {
            // Apply same pixel alignment to star points
            const x = Math.floor(padding + starPoints[i] * cellSize) + 0.5;
            const y = Math.floor(padding + starPoints[j] * cellSize) + 0.5;
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.fill();
        }
    }
}

function drawStones(stones) {
    console.log('Drawing stones:', stones);
    
    // Get the canvas size (it's a square now, so width = height)
    const canvasSize = canvas.width;
    
    // Calculate cell size (padding = cell size)
    const cellSize = canvasSize / (boardSize + 1);
    
    // Padding is equal to cellSize
    const padding = cellSize;
    
    // Calculate the actual grid size
    const gridSize = cellSize * (boardSize - 1);
    
    // Make stones slightly smaller than the cell size to prevent overlapping
    const stoneRadius = cellSize * 0.45;
    
    stones.forEach(stone => {
        // Apply same pixel alignment to stones
        const x = Math.floor(padding + stone.x * cellSize) + 0.5;
        const y = Math.floor(padding + stone.y * cellSize) + 0.5;
        
        ctx.beginPath();
        ctx.arc(x, y, stoneRadius, 0, 2 * Math.PI);
        ctx.fillStyle = stone.color;
        ctx.fill();
        ctx.strokeStyle = stone.color === 'black' ? '#000' : '#888';
        ctx.stroke();
    });
}

// Event handlers
async function loadSgfFiles() {
    const sgfSelect = document.getElementById('sgfFile');
    sgfSelect.innerHTML = '';
    
    sgfFiles.forEach(file => {
        const option = document.createElement('option');
        option.value = file.path;
        option.textContent = file.name;
        sgfSelect.appendChild(option);
    });
    
    if (sgfFiles.length > 0) {
        await loadSgfFile(sgfFiles[0].path);
    }
}

async function loadSgfFile(filename) {
    try {
        console.log('Loading SGF file:', filename);
        const response = await fetch(filename);
        if (!response.ok) {
            throw new Error(`Failed to load SGF file: ${response.statusText}`);
        }
        const sgfString = await response.text();
        currentSgf = parseSgf(sgfString);
        generateNewPosition();
    } catch (error) {
        console.error('Error loading SGF file:', error);
        alert('Failed to load SGF file. Please check the console for details.');
    }
}

function downloadSgf() {
    if (!currentBranch) return;
    
    // Convert current branch back to SGF format
    let sgfString = '(;';
    
    // Add root properties
    Object.entries(currentSgf.properties).forEach(([key, value]) => {
        sgfString += `${key}[${value}]`;
    });
    
    // Add moves
    currentBranch.moves.forEach(move => {
        const color = move.color === 'black' ? 'B' : 'W';
        const x = String.fromCharCode(move.x + 97);
        const y = String.fromCharCode(move.y + 97);
        sgfString += `;${color}[${x}${y}]`;
    });
    
    sgfString += ')';
    
    // Create and trigger download
    const blob = new Blob([sgfString], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'position.sgf';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Initialize canvas size
    resizeCanvas();
    
    loadSgfFiles();
    
    document.getElementById('newPosition').addEventListener('click', generateNewPosition);
    document.getElementById('downloadSgf').addEventListener('click', downloadSgf);
    document.getElementById('sgfFile').addEventListener('change', (e) => {
        loadSgfFile(e.target.value);
    });
    
    // Add window resize handler
    window.addEventListener('resize', resizeCanvas);
}); 