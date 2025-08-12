def print_solution(board, count):
    print(f"Solution {count}:")
    for row in board:
        print("  " + " ".join("Q" if col else "." for col in row))
    print()

def is_safe(board, row, col, n):
    for i in range(row):
        if board[i][col] or \
           (col - (row - i) >= 0 and board[i][col - (row - i)]) or \
           (col + (row - i) < n and board[i][col + (row - i)]):
            return False
    return True

def solve_n_queens_util(board, row, n, solutions):
    if row == n:
        solutions.append([r[:] for r in board])
        return
    for col in range(n):
        if is_safe(board, row, col, n):
            board[row][col] = 1
            solve_n_queens_util(board, row + 1, n, solutions)
            board[row][col] = 0

def solve_n_queens(n):
    board = [[0]*n for _ in range(n)]
    solutions = []
    solve_n_queens_util(board, 0, n, solutions)

    print(f"\n========== {n}-Queens ==========")
    print(f"Total solutions: {len(solutions)}\n")

    max_to_show = 2
    for idx, sol in enumerate(solutions[:max_to_show], 1):
        print_solution(sol, idx)

    if len(solutions) > max_to_show:
        print(f"...and {len(solutions) - max_to_show} more solutions not displayed.\n")

# Run the solver for n = 4 to 8
for n in range(4, 9):
    solve_n_queens(n)
