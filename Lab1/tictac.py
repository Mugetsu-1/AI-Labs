def show(b):
    print("\n+---+---+---+")
    for i in range(0, 9, 3):
        print(f"| {b[i]} | {b[i+1]} | {b[i+2]} |")
    print("+---+---+---+")

def valid(b, p):
    return 0 <= p <= 8 and b[p] == ' '

def move(b, p, s):
    if valid(b, p):
        b[p] = s
        return True
    return False

def win(b):
    w = [[0,1,2],[3,4,5],[6,7,8],
         [0,3,6],[1,4,7],[2,5,8],
         [0,4,8],[2,4,6]]
    for i in w:
        a, c, d = i
        if b[a] == b[c] == b[d] != ' ':
            return b[a]
    return None

def full(b):
    return ' ' not in b

def empty(b):
    return [i for i in range(9) if b[i] == ' ']

def ai(b, A, H):
    for p in empty(b):
        b[p] = A
        if win(b) == A:
            return
        b[p] = ' '
    for p in empty(b):
        b[p] = H
        if win(b) == H:
            b[p] = A
            return
        b[p] = ' '
    if valid(b, 4):
        move(b, 4, A)
        return
    for p in [0, 2, 6, 8]:
        if valid(b, p):
            move(b, p, A)
            return
    for p in empty(b):
        move(b, p, A)
        return

def human(b, H):
    while True:
        try:
            p = int(input("Your move (0-8): "))
            if move(b, p, H):
                return
            print("Invalid move.")
        except:
            print("Enter 0-8.")

def play():
    b = [' '] * 9
    A, H = 'X', 'O'
    print("\nTic-Tac-Toe: You = O, AI = X")
    print("Positions: 0|1|2\n           3|4|5\n           6|7|8")
    show(b)
    while True:
        ai(b, A, H)
        show(b)
        if win(b) == A:
            print("AI wins!")
            break
        if full(b):
            print("It's a tie!")
            break
        human(b, H)
        show(b)
        if win(b) == H:
            print("You win!")
            break
        if full(b):
            print("It's a tie!")
            break

def main():
    while True:
        play()
        if input("Play again? (y/n): ").lower() != 'y':
            print("Thanks for playing!")
            break

if __name__ == "__main__":
    main()

