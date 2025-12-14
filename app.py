import streamlit as st
import numpy as np
import time
from pathlib import Path

try:
    from model import HashiPuzzle
    from solvers import PySATSolver
    from utils import BridgeUtils
except ImportError:
    st.error("Không thể import các module cần thiết. Vui lòng đảm bảo các file model.py, solvers.py, logic.py, utils.py có trong cùng thư mục.")
    st.stop()

# Page config
st.set_page_config(
    page_title="Hashiwokakero Solver",
    page_icon="🌉",
    layout="wide"
)

# Helper functions
def display_interactive_puzzle(puzzle, user_bridges, selected_island=None):
    """Display interactive puzzle where user can play"""
    rows, cols = puzzle.grid.shape
    
    # Calculate cell size
    max_dimension = max(rows, cols)
    if max_dimension <= 7:
        cell_size = 50
    elif max_dimension <= 10:
        cell_size = 45
    elif max_dimension <= 15:
        cell_size = 40
    elif max_dimension <= 20:
        cell_size = 35
    else:
        cell_size = 30
    
    font_size = max(12, int(cell_size * 0.4))
    bridge_font_size = max(14, int(cell_size * 0.5))
    
    # Create output grid with bridges
    output = [['' for _ in range(cols)] for _ in range(rows)]
    
    # Place islands
    for r, c, val in puzzle.islands:
        output[r][c] = str(val)
    
    # Place user's bridges
    for (island1, island2), num_bridges in user_bridges.items():
        r1, c1 = island1
        r2, c2 = island2
        
        if r1 == r2:  # Horizontal
            symbol = '=' if num_bridges == 2 else '-'
            min_c, max_c = min(c1, c2), max(c1, c2)
            for c in range(min_c + 1, max_c):
                if output[r1][c] == '':
                    output[r1][c] = symbol
        else:  # Vertical
            symbol = '$' if num_bridges == 2 else '|'
            min_r, max_r = min(r1, r2), max(r1, r2)
            for r in range(min_r + 1, max_r):
                if output[r][c1] == '':
                    output[r][c1] = symbol
    
    # Create HTML with clickable islands
    html = f'''
    <div style="overflow-x: auto; overflow-y: auto; max-height: 80vh;">
        <table style="border-collapse: collapse; margin: auto; table-layout: fixed;">
    '''
    
    for i in range(rows):
        html += '<tr>'
        for j in range(cols):
            cell = output[i][j]
            if cell.isdigit():  # Island
                # Check if this island is selected
                is_selected = selected_island and selected_island == (i, j)
                bg_color = '#FF5722' if is_selected else '#4CAF50'
                
                # Calculate current bridges
                current_bridges = 0
                for (island1, island2), num_bridges in user_bridges.items():
                    if (i, j) == island1 or (i, j) == island2:
                        current_bridges += num_bridges
                
                # Show remaining bridges needed
                required = int(cell)
                remaining = required - current_bridges
                display_text = f"{cell}"
                if remaining != required:
                    display_text += f"<br><small>({remaining})</small>"
                
                html += f'''<td onclick="window.parent.postMessage({{type: 'island_click', row: {i}, col: {j}}}, '*')" 
                    style="
                    width: {cell_size}px; 
                    height: {cell_size}px;
                    min-width: {cell_size}px;
                    min-height: {cell_size}px;
                    max-width: {cell_size}px;
                    max-height: {cell_size}px;
                    text-align: center; 
                    vertical-align: middle;
                    border: 1px solid #ddd; 
                    background-color: {bg_color}; 
                    color: white; 
                    font-weight: bold; 
                    font-size: {font_size}px;
                    padding: 0;
                    box-sizing: border-box;
                    cursor: pointer;
                    transition: background-color 0.2s;
                ">{display_text}</td>'''
            elif cell in ['|', '$', '-', '=']:  # Bridge
                color = '#2196F3' if cell in ['|', '$'] else '#FF9800'
                html += f'''<td style="
                    width: {cell_size}px; 
                    height: {cell_size}px;
                    min-width: {cell_size}px;
                    min-height: {cell_size}px;
                    max-width: {cell_size}px;
                    max-height: {cell_size}px;
                    text-align: center; 
                    vertical-align: middle;
                    border: 1px solid #ddd; 
                    background-color: white; 
                    color: {color}; 
                    font-weight: bold; 
                    font-size: {bridge_font_size}px;
                    padding: 0;
                    box-sizing: border-box;
                ">{cell}</td>'''
            else:  # Empty
                html += f'''<td style="
                    width: {cell_size}px; 
                    height: {cell_size}px;
                    min-width: {cell_size}px;
                    min-height: {cell_size}px;
                    max-width: {cell_size}px;
                    max-height: {cell_size}px;
                    border: 1px solid #ddd; 
                    background-color: #f9f9f9;
                    padding: 0;
                    box-sizing: border-box;
                "></td>'''
        html += '</tr>'
    
    html += '''</table></div>
    <script>
        window.addEventListener('message', function(e) {
            if (e.data.type === 'island_click') {
                // Forward to Streamlit
                window.parent.postMessage(e.data, '*');
            }
        });
    </script>
    '''
    
    return html

def display_puzzle(puzzle):
    """Display the initial puzzle grid (non-interactive)"""
    rows, cols = puzzle.grid.shape
    
    max_dimension = max(rows, cols)
    if max_dimension <= 7:
        cell_size = 50
    elif max_dimension <= 10:
        cell_size = 45
    elif max_dimension <= 15:
        cell_size = 40
    elif max_dimension <= 20:
        cell_size = 35
    else:
        cell_size = 30
    
    font_size = max(12, int(cell_size * 0.4))
    
    html = f'''
    <div style="overflow-x: auto; overflow-y: auto; max-height: 80vh;">
        <table style="border-collapse: collapse; margin: auto; table-layout: fixed;">
    '''
    
    for i in range(rows):
        html += '<tr>'
        for j in range(cols):
            val = puzzle.grid[i][j]
            if val > 0:
                html += f'''<td style="
                    width: {cell_size}px; 
                    height: {cell_size}px; 
                    min-width: {cell_size}px;
                    min-height: {cell_size}px;
                    max-width: {cell_size}px;
                    max-height: {cell_size}px;
                    text-align: center; 
                    vertical-align: middle;
                    border: 1px solid #ddd; 
                    background-color: #4CAF50; 
                    color: white; 
                    font-weight: bold; 
                    font-size: {font_size}px;
                    padding: 0;
                    box-sizing: border-box;
                ">{val}</td>'''
            else:
                html += f'''<td style="
                    width: {cell_size}px; 
                    height: {cell_size}px;
                    min-width: {cell_size}px;
                    min-height: {cell_size}px;
                    max-width: {cell_size}px;
                    max-height: {cell_size}px;
                    border: 1px solid #ddd; 
                    background-color: #f9f9f9;
                    padding: 0;
                    box-sizing: border-box;
                "></td>'''
        html += '</tr>'
    
    html += '</table></div>'
    return html

def display_solution(puzzle, solution):
    """Display the solved puzzle with bridges"""
    rows, cols = puzzle.rows, puzzle.cols
    output = [['' for _ in range(cols)] for _ in range(rows)]
    
    for r, c, val in puzzle.islands:
        output[r][c] = str(val)
    
    for (island1, island2), num_bridges in solution.items():
        r1, c1 = island1
        r2, c2 = island2
        
        if r1 == r2:
            symbol = '=' if num_bridges == 2 else '-'
            min_c, max_c = min(c1, c2), max(c1, c2)
            for c in range(min_c + 1, max_c):
                if output[r1][c] == '':
                    output[r1][c] = symbol
        else:
            symbol = '$' if num_bridges == 2 else '|'
            min_r, max_r = min(r1, r2), max(r1, r2)
            for r in range(min_r + 1, max_r):
                if output[r][c1] == '':
                    output[r][c1] = symbol
    
    max_dimension = max(rows, cols)
    if max_dimension <= 7:
        cell_size = 50
    elif max_dimension <= 10:
        cell_size = 45
    elif max_dimension <= 15:
        cell_size = 40
    elif max_dimension <= 20:
        cell_size = 35
    else:
        cell_size = 30
    
    font_size = max(12, int(cell_size * 0.4))
    bridge_font_size = max(14, int(cell_size * 0.5))
    
    html = f'''
    <div style="overflow-x: auto; overflow-y: auto; max-height: 80vh;">
        <table style="border-collapse: collapse; margin: auto; table-layout: fixed;">
    '''
    
    for i in range(rows):
        html += '<tr>'
        for j in range(cols):
            cell = output[i][j]
            if cell.isdigit():
                html += f'''<td style="
                    width: {cell_size}px; 
                    height: {cell_size}px;
                    min-width: {cell_size}px;
                    min-height: {cell_size}px;
                    max-width: {cell_size}px;
                    max-height: {cell_size}px;
                    text-align: center; 
                    vertical-align: middle;
                    border: 1px solid #ddd; 
                    background-color: #4CAF50; 
                    color: white; 
                    font-weight: bold; 
                    font-size: {font_size}px;
                    padding: 0;
                    box-sizing: border-box;
                ">{cell}</td>'''
            elif cell in ['|', '$', '-', '=']:
                color = '#2196F3' if cell in ['|', '$'] else '#FF9800'
                html += f'''<td style="
                    width: {cell_size}px; 
                    height: {cell_size}px;
                    min-width: {cell_size}px;
                    min-height: {cell_size}px;
                    max-width: {cell_size}px;
                    max-height: {cell_size}px;
                    text-align: center; 
                    vertical-align: middle;
                    border: 1px solid #ddd; 
                    background-color: white; 
                    color: {color}; 
                    font-weight: bold; 
                    font-size: {bridge_font_size}px;
                    padding: 0;
                    box-sizing: border-box;
                ">{cell}</td>'''
            else:
                html += f'''<td style="
                    width: {cell_size}px; 
                    height: {cell_size}px;
                    min-width: {cell_size}px;
                    min-height: {cell_size}px;
                    max-width: {cell_size}px;
                    max-height: {cell_size}px;
                    border: 1px solid #ddd; 
                    background-color: #f9f9f9;
                    padding: 0;
                    box-sizing: border-box;
                "></td>'''
        html += '</tr>'
    
    html += '</table></div>'
    return html

def check_user_solution(puzzle, user_bridges):
    """Check if user's solution is valid"""
    errors = []
    
    # Check each island's bridge count
    for r, c, required in puzzle.islands:
        island_pos = (r, c)
        total_bridges = 0
        
        for (island1, island2), num_bridges in user_bridges.items():
            if island_pos == island1 or island_pos == island2:
                total_bridges += num_bridges
        
        if total_bridges != required:
            errors.append(f"Đảo ({r},{c}): cần {required} cầu, hiện có {total_bridges}")
    
    # Check connectivity
    if not BridgeUtils.is_connected(user_bridges, puzzle.islands):
        errors.append("Các đảo chưa được kết nối với nhau!")
    
    return len(errors) == 0, errors

def can_add_bridge(puzzle, island1, island2, user_bridges):
    """Check if a bridge can be added between two islands"""
    r1, c1 = island1
    r2, c2 = island2
    
    # Must be in same row or column
    if r1 != r2 and c1 != c2:
        return False
    
    # Check if islands are neighbors (no island in between)
    if r1 == r2:  # Horizontal
        min_c, max_c = min(c1, c2), max(c1, c2)
        for c in range(min_c + 1, max_c):
            if puzzle.grid[r1][c] > 0:
                return False
    else:  # Vertical
        min_r, max_r = min(r1, r2), max(r1, r2)
        for r in range(min_r + 1, max_r):
            if puzzle.grid[r][c1] > 0:
                return False
    
    # Check for crossing bridges
    new_bridge = tuple(sorted([island1, island2]))
    for existing_bridge in user_bridges.keys():
        if BridgeUtils.bridges_cross(new_bridge, existing_bridge):
            return False
    
    return True

def get_island_stats(puzzle):
    """Get statistics about islands by degree"""
    island_counts = {}
    for r, c, val in puzzle.islands:
        island_counts[val] = island_counts.get(val, 0) + 1
    return island_counts

def get_bridge_stats(solution):
    """Get statistics about bridges by type"""
    bridge_counts = {'horizontal_single': 0, 'horizontal_double': 0, 
                    'vertical_single': 0, 'vertical_double': 0}
    
    for (island1, island2), num_bridges in solution.items():
        r1, c1 = island1
        r2, c2 = island2
        
        if r1 == r2:
            if num_bridges == 1:
                bridge_counts['horizontal_single'] += 1
            else:
                bridge_counts['horizontal_double'] += 1
        else:
            if num_bridges == 1:
                bridge_counts['vertical_single'] += 1
            else:
                bridge_counts['vertical_double'] += 1
    
    return bridge_counts

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1em;
        padding: 0.6em;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1557a0;
    }
    
    .info-icon {
        display: inline-block;
        width: 18px;
        height: 18px;
        background-color: #1f77b4;
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 18px;
        font-size: 12px;
        font-weight: bold;
        cursor: help;
        margin-left: 5px;
        position: relative;
    }
    
    .info-icon .tooltip {
        visibility: hidden;
        width: 220px;
        background-color: #333;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -110px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 13px;
        line-height: 1.5;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .info-icon .tooltip::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #333 transparent transparent transparent;
    }
    
    .info-icon:hover .tooltip {
        visibility: visible;
        opacity: 1;
    }
    
    .stat-label {
        display: flex;
        align-items: center;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'puzzle' not in st.session_state:
    st.session_state.puzzle = None
if 'solution' not in st.session_state:
    st.session_state.solution = None
if 'solving' not in st.session_state:
    st.session_state.solving = False
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'user_bridges' not in st.session_state:
    st.session_state.user_bridges = {}
if 'selected_island' not in st.session_state:
    st.session_state.selected_island = None
if 'game_mode' not in st.session_state:
    st.session_state.game_mode = 'play'  # 'play' or 'solve'

# Header
st.markdown('<div class="main-header">🌉 Hashiwokakero Solver</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Chơi hoặc giải bài toán nối cầu sử dụng PySAT Solver</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Cài đặt")
    
    # Mode selection
    mode = st.radio(
        "Chế độ:",
        options=['🎮 Chơi', '🤖 Giải tự động'],
        index=0 if st.session_state.game_mode == 'play' else 1
    )
    st.session_state.game_mode = 'play' if '🎮' in mode else 'solve'
    
    st.markdown("---")
    
    # Size selection
    size_options = {
        "4x4": 1, "5x5": 2, "6x6": 3, "7x7": 4, "8x8": 5,
        "9x9": 6, "10x10": 7, "11x11": 8, "12x12": 9, "13x13": 10,
        "14x14": 11, "15x15": 12, "16x16": 13, "17x17": 14, "18x18": 15,
        "19x19": 16, "20x20": 17, "21x21": 18, "22x22": 19, "23x23": 20,
        "24x24": 21, "25x25": 22
    }
    
    selected_size = st.selectbox(
        "Chọn kích thước puzzle:",
        options=list(size_options.keys()),
        index=0
    )
    
    file_number = size_options[selected_size]
    input_file = f"inputs/input-{file_number:02d}.txt"
    
    st.info(f"📁 File: `{input_file}`")
    
    # Load puzzle button
    if st.button("📥 Tải Puzzle", use_container_width=True):
        try:
            if not Path(input_file).exists():
                st.error(f"❌ File {input_file} không tồn tại!")
            else:
                st.session_state.puzzle = HashiPuzzle.read_from_file(input_file)
                st.session_state.solution = None
                st.session_state.stats = None
                st.session_state.user_bridges = {}
                st.session_state.selected_island = None
                st.success(f"✅ Đã tải puzzle {selected_size}")
        except Exception as e:
            st.error(f"❌ Lỗi khi tải file: {str(e)}")
    
    st.markdown("---")
    
    if st.session_state.game_mode == 'play' and st.session_state.puzzle:
        st.subheader("🎮 Điều khiển")
        st.markdown("""
        **Cách chơi:**
        1. Click vào đảo đầu tiên
        2. Click vào đảo thứ hai để nối cầu
        3. Click lại để tăng cầu (1→2→xóa)
        
        **Số trong ngoặc:** cầu còn thiếu
        """)
    
    elif st.session_state.game_mode == 'solve':
        # Solve button
        if st.button("🚀 Giải Puzzle", use_container_width=True, disabled=st.session_state.puzzle is None):
            if st.session_state.puzzle:
                st.session_state.solving = True
    
    # Thoát khỏi puzzle (chỉ hiện khi đã load puzzle)
    if st.session_state.puzzle:
        if st.button("⬅️ Thoát", use_container_width=True):
            st.session_state.puzzle = None
            st.session_state.solution = None
            st.session_state.solving = False
            st.session_state.stats = None
            st.session_state.user_bridges = {}
            st.session_state.selected_island = None
            st.rerun()
    
    st.markdown("---")
    
    # Instructions
    with st.expander("📖 Hướng dẫn"):
        st.markdown("""
        **Ký hiệu:**
        - Số: Đảo (island)
        - `-`: Cầu ngang đơn
        - `=`: Cầu ngang kép
        - `|`: Cầu dọc đơn
        - `$`: Cầu dọc kép
        """)
    
    with st.expander("ℹ️ Về Hashiwokakero"):
        st.markdown("""
        Hashiwokakero là trò chơi logic Nhật Bản 
        với mục tiêu nối các đảo bằng cầu theo 
        các quy tắc:
        - Cầu phải thẳng hàng
        - Không được giao nhau
        - Tối đa 2 cầu giữa 2 đảo
        - Số cầu = số trên đảo
        - Tất cả đảo phải liên thông
        """)

# Main content
if st.session_state.puzzle is None:
    st.info("👈 Vui lòng chọn kích thước và tải puzzle từ sidebar")
else:
    if st.session_state.game_mode == 'play':
        # Play mode - use React component
        st.subheader("🎮 Chơi Puzzle")
        
        # Convert puzzle to JSON format for React
        puzzle_data = {
            'grid': st.session_state.puzzle.grid.tolist()
        }
        
        st.info("💡 **Cách chơi:** Click vào đảo đầu tiên (màu đỏ), sau đó click vào đảo thứ hai để nối cầu. Click nhiều lần: 0 → 1 cầu → 2 cầu → xóa")
        
        # Embed React component
        react_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <style>
        body {{ margin: 0; padding: 20px; font-family: system-ui, -apple-system, sans-serif; }}
        * {{ box-sizing: border-box; }}
    </style>
</head>
<body>
    <div id="root"></div>
    <script type="text/babel">
        const {{ useState, useEffect }} = React;

        const HashiGame = () => {{
            const initialGrid = {puzzle_data['grid']};
            const [grid] = useState(initialGrid);
            const [bridges, setBridges] = useState({{}});
            const [selectedIsland, setSelectedIsland] = useState(null);
            const [errors, setErrors] = useState([]);
            const [showSuccess, setShowSuccess] = useState(false);
            const [moveCount, setMoveCount] = useState(0);

            // THÊM TIMER
            const [startTime, setStartTime] = useState(Date.now());
            const [elapsedTime, setElapsedTime] = useState(0);
            
            // Undo, Redo
            const [history, setHistory] = useState([{{}}]); // Lưu lịch sử bridges
            const [historyIndex, setHistoryIndex] = useState(0); // Vị trí hiện tại trong history

            // Cập nhật timer mỗi giây
            useEffect(() => {{
                const interval = setInterval(() => {{
                    setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
                }}, 1000);
                return () => clearInterval(interval);
            }}, [startTime]);
            
            useEffect(() => {{
                const handleKeyDown = (e) => {{
                    // Ctrl+Z hoặc Cmd+Z cho Undo
                    if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {{
                        e.preventDefault();
                        undo();
                    }}
                    // Ctrl+Y hoặc Cmd+Shift+Z cho Redo
                    if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.shiftKey && e.key === 'z'))) {{
                        e.preventDefault();
                        redo();
                    }}
                }};
                
                window.addEventListener('keydown', handleKeyDown);
                return () => window.removeEventListener('keydown', handleKeyDown);
            }}, [historyIndex, history]);

            // Hàm format thời gian MM:SS
            const formatTime = (seconds) => {{
                const mins = Math.floor(seconds / 60);
                const secs = seconds % 60;
                return `${{mins}}:${{secs.toString().padStart(2, '0')}}`;
            }};

            const getIslands = () => {{
                const islands = [];
                grid.forEach((row, r) => {{
                    row.forEach((val, c) => {{
                        if (val > 0) islands.push({{ r, c, val }});
                    }});
                }});
                return islands;
            }};

            const isSamePosition = (pos1, pos2) => {{
                return pos1.r === pos2.r && pos1.c === pos2.c;
            }};

            const getBridgeKey = (island1, island2) => {{
                const pos1 = `${{island1.r}},${{island1.c}}`;
                const pos2 = `${{island2.r}},${{island2.c}}`;
                return pos1 < pos2 ? `${{pos1}}-${{pos2}}` : `${{pos2}}-${{pos1}}`;
            }};

            const canConnect = (island1, island2) => {{
                if (island1.r !== island2.r && island1.c !== island2.c) return false;

                if (island1.r === island2.r) {{
                    const minC = Math.min(island1.c, island2.c);
                    const maxC = Math.max(island1.c, island2.c);
                    for (let c = minC + 1; c < maxC; c++) {{
                        if (grid[island1.r][c] > 0) return false;
                    }}
                }} else {{
                    const minR = Math.min(island1.r, island2.r);
                    const maxR = Math.max(island1.r, island2.r);
                    for (let r = minR + 1; r < maxR; r++) {{
                        if (grid[r][island1.c] > 0) return false;
                    }}
                }}

                const newBridge = {{ island1, island2 }};
                for (const [key, count] of Object.entries(bridges)) {{
                    if (count === 0) continue;
                    const [pos1, pos2] = key.split('-');
                    const [r1, c1] = pos1.split(',').map(Number);
                    const [r2, c2] = pos2.split(',').map(Number);
                    const existingBridge = {{
                        island1: {{ r: r1, c: c1 }},
                        island2: {{ r: r2, c: c2 }}
                    }};
                    if (bridgesCross(newBridge, existingBridge)) return false;
                }}
                return true;
            }};

            const bridgesCross = (bridge1, bridge2) => {{
                const {{ island1: i1, island2: i2 }} = bridge1;
                const {{ island1: i3, island2: i4 }} = bridge2;

                if (i1.r === i2.r && i3.c === i4.c) {{
                    const minC = Math.min(i1.c, i2.c);
                    const maxC = Math.max(i1.c, i2.c);
                    const minR = Math.min(i3.r, i4.r);
                    const maxR = Math.max(i3.r, i4.r);
                    if (minC < i3.c && i3.c < maxC && minR < i1.r && i1.r < maxR) return true;
                }}

                if (i1.c === i2.c && i3.r === i4.r) {{
                    const minR = Math.min(i1.r, i2.r);
                    const maxR = Math.max(i1.r, i2.r);
                    const minC = Math.min(i3.c, i4.c);
                    const maxC = Math.max(i3.c, i4.c);
                    if (minR < i3.r && i3.r < maxR && minC < i1.c && i1.c < maxC) return true;
                }}
                return false;
            }};

            const handleIslandClick = (r, c) => {{
                const clickedIsland = {{ r, c }};
                if (!selectedIsland) {{
                    setSelectedIsland(clickedIsland);
                    setErrors([]);
                }} else if (isSamePosition(selectedIsland, clickedIsland)) {{
                    setSelectedIsland(null);
                }} else {{
                    if (canConnect(selectedIsland, clickedIsland)) {{
                        const key = getBridgeKey(selectedIsland, clickedIsland);
                        const current = bridges[key] || 0;
                        const island1Count = getBridgeCount(selectedIsland.r, selectedIsland.c);
                        const island2Count = getBridgeCount(clickedIsland.r, clickedIsland.c);
                        const island1Max = grid[selectedIsland.r][selectedIsland.c];
                        const island2Max = grid[clickedIsland.r][clickedIsland.c];
                        const newBridges = {{ ...bridges }};

                        if (current === 0) {{
                            if (island1Count < island1Max && island2Count < island2Max) {{
                                newBridges[key] = 1;
                                const newHistory = history.slice(0, historyIndex + 1);
                                newHistory.push(newBridges);
                                setHistory(newHistory);
                                setHistoryIndex(historyIndex + 1);
                                setBridges(newBridges);
                                setMoveCount(moveCount + 1);
                            }} else {{
                                setErrors(['Một trong hai đảo đã đủ số cầu!']);
                            }}
                        }} else if (current === 1) {{
                            if (island1Count < island1Max && island2Count < island2Max) {{
                                newBridges[key] = 2;
                                const newHistory = history.slice(0, historyIndex + 1);
                                newHistory.push(newBridges);
                                setHistory(newHistory);
                                setHistoryIndex(historyIndex + 1);
                                setBridges(newBridges);
                                setMoveCount(moveCount + 1);
                            }} else {{
                                setErrors(['Một trong hai đảo đã đủ số cầu!']);
                            }}
                        }} else {{
                            newBridges[key] = 0;
                            const newHistory = history.slice(0, historyIndex + 1);
                            newHistory.push(newBridges);
                            setHistory(newHistory);
                            setHistoryIndex(historyIndex + 1);
                            setBridges(newBridges);
                            setMoveCount(moveCount + 1);
                        }}
                    }} else {{
                        setErrors(['Không thể nối hai đảo này!']);
                    }}
                    setSelectedIsland(null);
                }}
            }};

            const getBridgeCount = (r, c) => {{
                let count = 0;
                Object.entries(bridges).forEach(([key, num]) => {{
                    if (num === 0) return;
                    const [pos1, pos2] = key.split('-');
                    const [r1, c1] = pos1.split(',').map(Number);
                    const [r2, c2] = pos2.split(',').map(Number);
                    if ((r === r1 && c === c1) || (r === r2 && c === c2)) count += num;
                }});
                return count;
            }};

            const checkSolution = () => {{
                const islands = getIslands();
                let wrongIslands = 0;
                islands.forEach(({{ r, c, val }}) => {{
                    const count = getBridgeCount(r, c);
                    if (count !== val) wrongIslands++;
                }});
                
                const connected = isConnected();
                
                if (wrongIslands === 0 && connected) {{
                    setShowSuccess(true);
                    setTimeout(() => setShowSuccess(false), 3000);
                    setErrors([]);
                }} else {{
                    const errorMsg = [];
                    if (wrongIslands > 0) errorMsg.push(`${{wrongIslands}} đảo chưa đủ/thừa cầu`);
                    if (!connected) errorMsg.push('các đảo chưa kết nối với nhau.');
                    setErrors(errorMsg);
                }}
            }};

            const isConnected = () => {{
                const islands = getIslands();
                if (islands.length === 0) return true;
                const adj = {{}};
                islands.forEach(({{ r, c }}) => {{ adj[`${{r}},${{c}}`] = []; }});
                Object.entries(bridges).forEach(([key, num]) => {{
                    if (num === 0) return;
                    const [pos1, pos2] = key.split('-');
                    adj[pos1].push(pos2);
                    adj[pos2].push(pos1);
                }});
                const start = `${{islands[0].r}},${{islands[0].c}}`;
                const visited = new Set([start]);
                const queue = [start];
                while (queue.length > 0) {{
                    const current = queue.shift();
                    adj[current].forEach(neighbor => {{
                        if (!visited.has(neighbor)) {{
                            visited.add(neighbor);
                            queue.push(neighbor);
                        }}
                    }});
                }}
                return visited.size === islands.length;
            }};

            const resetGame = () => {{
                setBridges({{}});
                setSelectedIsland(null);
                setErrors([]);
                setShowSuccess(false);
                setMoveCount(0);
                setStartTime(Date.now());     // Reset timer
                setElapsedTime(0);            // Reset hiển thị về 0:00
                setHistory([{{}}]);
                setElapsedTime(0);
            }};
            
            const undo = () => {{
                if (historyIndex > 0) {{
                    const newIndex = historyIndex - 1;
                    setHistoryIndex(newIndex);
                    setBridges(history[newIndex]);
                    setErrors([]);
                }}
            }};

            const redo = () => {{
                if (historyIndex < history.length - 1) {{
                    const newIndex = historyIndex + 1;
                    setHistoryIndex(newIndex);
                    setBridges(history[newIndex]);
                    setErrors([]);
                }}
            }};

            const renderCell = (r, c) => {{
                const val = grid[r][c];
                const isIsland = val > 0;
                const isSelected = selectedIsland && isSamePosition(selectedIsland, {{ r, c }});
                
                if (isIsland) {{
                    const current = getBridgeCount(r, c);
                    const remaining = val - current;
                    const isComplete = remaining === 0;
                    
                    return React.createElement('div', {{
                        onClick: () => handleIslandClick(r, c),
                        style: {{
                            width: '48px',
                            height: '48px',
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            cursor: 'pointer',
                            fontWeight: 'bold',
                            color: 'white',
                            borderRadius: '4px',
                            transition: 'all 0.2s',
                            backgroundColor: isSelected ? '#EF4444' : isComplete ? '#16A34A' : '#3B82F6',
                            transform: isSelected ? 'scale(1.1)' : 'scale(1)',
                            boxShadow: isSelected ? '0 4px 6px rgba(0,0,0,0.1)' : 'none'
                        }}
                    }},
                        React.createElement('div', {{ style: {{ fontSize: '20px' }} }}, val),
                        !isComplete && React.createElement('div', {{ style: {{ fontSize: '12px', opacity: 0.75 }} }}, `(${{remaining}})`)
                    );
                }}

                let bridgeDisplay = null;
                Object.entries(bridges).forEach(([key, num]) => {{
                    if (num === 0) return;
                    const [pos1, pos2] = key.split('-');
                    const [r1, c1] = pos1.split(',').map(Number);
                    const [r2, c2] = pos2.split(',').map(Number);

                    if (r1 === r2 && r1 === r) {{
                        const minC = Math.min(c1, c2);
                        const maxC = Math.max(c1, c2);
                        if (c > minC && c < maxC) {{
                            bridgeDisplay = React.createElement('div', {{
                                style: {{
                                    width: '48px',
                                    height: '48px',
                                    display: 'flex',
                                    flexDirection: 'column',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    gap: num === 2 ? '4px' : '0',
                                    backgroundColor: '#F3F4F6'
                                }}
                            }},
                                React.createElement('div', {{ style: {{ width: '100%', height: num === 1 ? '4px' : '2px', backgroundColor: '#F59E0B' }} }}),
                                num === 2 && React.createElement('div', {{ style: {{ width: '100%', height: '2px', backgroundColor: '#F59E0B' }} }})
                            );
                        }}
                    }}

                    if (c1 === c2 && c1 === c) {{
                        const minR = Math.min(r1, r2);
                        const maxR = Math.max(r1, r2);
                        if (r > minR && r < maxR) {{
                            bridgeDisplay = React.createElement('div', {{
                                style: {{
                                    width: '48px',
                                    height: '48px',
                                    display: 'flex',
                                    flexDirection: 'row',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    gap: num === 2 ? '4px' : '0',
                                    backgroundColor: '#F3F4F6'
                                }}
                            }},
                                React.createElement('div', {{ style: {{ width: num === 1 ? '4px' : '2px', height: '100%', backgroundColor: '#F59E0B' }} }}),
                                num === 2 && React.createElement('div', {{ style: {{ width: '2px', height: '100%', backgroundColor: '#F59E0B' }} }})
                            );
                        }}
                    }}
                }});

                return bridgeDisplay || React.createElement('div', {{
                    style: {{ width: '48px', height: '48px', backgroundColor: '#F3F4F6' }}
                }});
            }};

            return React.createElement('div', {{ style: {{ padding: '20px', display: 'flex', flexDirection: 'column', alignItems: 'center' }} }},
                React.createElement('div', {{ style: {{ display: 'flex', gap: '10px', justifyContent: 'center', marginBottom: '10px', flexWrap: 'wrap' }} }},
                    React.createElement('button', {{
                        onClick: undo,
                        disabled: historyIndex === 0,
                        style: {{
                            padding: '10px 20px',
                            backgroundColor: historyIndex === 0 ? '#D1D5DB' : '#8B5CF6',
                            color: 'white',
                            border: 'none',
                            borderRadius: '6px',
                            cursor: historyIndex === 0 ? 'not-allowed' : 'pointer',
                            fontWeight: 'bold'
                        }}
                    }}, '↩️ Undo (Ctrl+Z)'),
                    React.createElement('button', {{
                        onClick: redo,
                        disabled: historyIndex >= history.length - 1,
                        style: {{
                            padding: '10px 20px',
                            backgroundColor: historyIndex >= history.length - 1 ? '#D1D5DB' : '#8B5CF6',
                            color: 'white',
                            border: 'none',
                            borderRadius: '6px',
                            cursor: historyIndex >= history.length - 1 ? 'not-allowed' : 'pointer',
                            fontWeight: 'bold'
                        }}
                    }}, '↪️ Redo (Ctrl+Y)'),
                    React.createElement('button', {{
                        onClick: checkSolution,
                        style: {{ padding: '10px 20px', backgroundColor: '#10B981', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold' }}
                    }}, '✅ Kiểm tra'),
                    React.createElement('button', {{
                        onClick: resetGame,
                        style: {{ padding: '10px 20px', backgroundColor: '#6B7280', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold' }}
                    }}, '🔄 Reset'),
                ),
                React.createElement('div', {{ style: {{ display: 'flex', gap: '10px', justifyContent: 'center', flexWrap: 'wrap', marginBottom: '20px' }} }},
                    React.createElement('div', {{
                        style: {{ padding: '10px 20px', backgroundColor: '#EFF6FF', borderRadius: '6px', fontWeight: 'bold' }}
                    }}, `Số nước đi: ${{moveCount}}`),
                    React.createElement('div', {{
                        style: {{ padding: '10px 20px', backgroundColor: '#FEF3C7', borderRadius: '6px', fontWeight: 'bold' }}
                    }}, `⏱️ Thời gian: ${{formatTime(elapsedTime)}}`)
                ),
                showSuccess && React.createElement('div', {{
                    style: {{ padding: '15px', backgroundColor: '#D1FAE5', border: '2px solid #10B981', borderRadius: '8px', marginBottom: '20px', textAlign: 'center', fontWeight: 'bold', color: '#065F46' }}
                }}, '🎉 Chúc mừng! Bạn đã giải đúng puzzle!'),
                errors.length > 0 && !showSuccess && React.createElement('div', {{
                    style: {{
                        padding: '15px',
                        backgroundColor: '#FEF3C7',
                        border: '2px solid #F59E0B',
                        borderRadius: '8px',
                        marginBottom: '20px',
                        textAlign: 'center'
                    }}
                }},
                    React.createElement('p', {{ style: {{ fontWeight: 'bold', color: '#92400E' }} }}, 
                        `Chưa hoàn thành: ${{errors.join(', ')}}`
                    )
                ),
                React.createElement('div', {{
                    style: {{ display: 'inline-block', border: '4px solid #C7D2FE', borderRadius: '8px', overflow: 'hidden' }}
                }},
                    grid.map((row, r) => React.createElement('div', {{ key: r, style: {{ display: 'flex' }} }},
                        row.map((_, c) => React.createElement('div', {{
                            key: c,
                            style: {{ border: '1px solid #E5E7EB' }}
                        }}, renderCell(r, c)))
                    ))
                )
            );
        }};

        ReactDOM.render(React.createElement(HashiGame), document.getElementById('root'));
    </script>
</body>
</html>
        """
        
        st.components.v1.html(react_html, height=800, scrolling=True)
    
    else:
        # Solve mode - show before/after
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Puzzle Ban Đầu")
            st.markdown(display_puzzle(st.session_state.puzzle), unsafe_allow_html=True)
        
        with col2:
            st.subheader("✨ Kết Quả")
            if st.session_state.solution:
                st.markdown(display_solution(st.session_state.puzzle, st.session_state.solution), unsafe_allow_html=True)
            else:
                st.info("Chưa có lời giải. Nhấn 'Giải Puzzle' để bắt đầu.")
        
        # Stats display
        if st.session_state.stats:
            st.markdown("---")
            st.subheader("📊 Thống Kê")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("⏱️ Thời gian giải", f"{st.session_state.stats['solve_time']:.3f}s")
            
            with col2:
                st.metric("🔢 Số biến", st.session_state.stats['num_variables'])
            
            with col3:
                st.metric("📝 Số mệnh đề", st.session_state.stats['num_clauses'])
            
            with col4:
                island_stats = get_island_stats(st.session_state.puzzle)
                island_tooltip = "Chi tiết theo bậc đảo:<br>"
                for degree in sorted(island_stats.keys()):
                    island_tooltip += f"• Bậc {degree}: {island_stats[degree]} đảo<br>"
                
                st.markdown(f"""
                    <div class="stat-label">
                        <div style="text-align: center; width: 100%;">
                            <div style="color: white; font-size: 0.875rem;">🏝️ Số đảo
                                <span class="info-icon">i
                                    <span class="tooltip">{island_tooltip}</span>
                                </span>
                            </div>
                            <div style="color: white; font-size: 2.25rem; font-weight: 600;">
                                {st.session_state.stats['num_islands']}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col5:
                bridge_stats = get_bridge_stats(st.session_state.solution)
                total_bridges = sum(bridge_stats.values())
                bridge_tooltip = "Chi tiết theo loại cầu:<br>"
                bridge_tooltip += f"• Ngang đơn (-): {bridge_stats['horizontal_single']}<br>"
                bridge_tooltip += f"• Ngang kép (=): {bridge_stats['horizontal_double']}<br>"
                bridge_tooltip += f"• Dọc đơn (|): {bridge_stats['vertical_single']}<br>"
                bridge_tooltip += f"• Dọc kép ($): {bridge_stats['vertical_double']}"
                
                st.markdown(f"""
                    <div class="stat-label">
                        <div style="text-align: center; width: 100%;">
                            <div style="color: white; font-size: 0.875rem;">🌉 Số cầu
                                <span class="info-icon">i
                                    <span class="tooltip">{bridge_tooltip}</span>
                                </span>
                            </div>
                            <div style="color: white; font-size: 2.25rem; font-weight: 600;">
                                {total_bridges}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# Solve logic
if st.session_state.solving and st.session_state.puzzle:
    with st.spinner("🔍 Đang giải puzzle..."):
        try:
            solver = PySATSolver(st.session_state.puzzle)
            success, solution = solver.solve()
            
            if success:
                st.session_state.solution = solution
                st.session_state.stats = solver.get_stats()
                st.success(f"✅ Đã tìm được lời giải trong {solver.solve_time:.3f}s!")
            else:
                st.error("❌ Không tìm thấy lời giải hợp lệ!")
                st.session_state.solution = None
                st.session_state.stats = None
        except Exception as e:
            st.error(f"❌ Lỗi khi giải: {str(e)}")
            st.session_state.solution = None
    
    st.session_state.solving = False
    st.rerun()