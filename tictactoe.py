
def check_winner(board):
    """
    Vérifie si un joueur a gagné.

    Entrée :
        board : matrice 3x3

    Sortie :
        "square", "cross" ou None
    """

    # Vérification des lignes
    for row in range(3):
        if (board[row][0] is not None
                and board[row][0] == board[row][1] == board[row][2]):
            return board[row][0]

    # Vérification des colonnes
    for col in range(3):
        if (board[0][col] is not None
                and board[0][col] == board[1][col] == board[2][col]):
            return board[0][col]

    # Diagonale principale
    if (board[0][0] is not None
            and board[0][0] == board[1][1] == board[2][2]):
        return board[0][0]

    # Diagonale secondaire
    if (board[0][2] is not None
            and board[0][2] == board[1][1] == board[2][0]):
        return board[0][2]

    return None


def minimax(board, maximizing, depth=0):
    """
    Algorithme classique Minimax appliqué au morpion 3x3.

    Principe :
        Explorer les coups possibles en supposant que
        les deux joueurs adoptent une stratégie optimale.

    Implémentation générée avec l'assistance d'OpenAI.

    Référence :
        Harvard CS50 AI — Tic-Tac-Toe
        https://cs50.harvard.edu/ai/projects/0/tictactoe/

    Convention :
        square : joueur maximisant
        cross  : joueur minimisant
    """

    winner = check_winner(board)

    # Conditions de fin
    if winner == "square":
        return 10 - depth

    if winner == "cross":
        return depth - 10

    if all(cell is not None for row in board for cell in row):
        return 0

    # Tour des carrés : maximiser le score
    if maximizing:
        best_score = -float("inf")
        symbol = "square"

    # Tour des croix : minimiser le score
    else:
        best_score = float("inf")
        symbol = "cross"

    # Tester tous les coups disponibles
    for row in range(3):
        for col in range(3):

            if board[row][col] is None:

                # Jouer temporairement
                board[row][col] = symbol

                # Explorer la suite de la partie
                score = minimax(board, not maximizing, depth + 1)

                # Annuler le coup
                board[row][col] = None

                # Conserver le meilleur score
                if maximizing:
                    best_score = max(best_score, score)
                else:
                    best_score = min(best_score, score)

    return best_score


def find_best_move(board):
    """
    Détermine le meilleur coup pour les carrés.

    Entrée :
        board : matrice 3x3 contenant
                "square", "cross" ou None

    Sortie :
        best_move : (row, col)
                    ou None si la partie est terminée
    """

    # Vérifier si la partie est terminée
    if check_winner(board) is not None:
        return None

    best_score = -float("inf")
    best_move = None

    # Tester chaque case vide
    for row in range(3):
        for col in range(3):

            if board[row][col] is None:

                # Simuler un carré
                board[row][col] = "square"

                # L'adversaire joue ensuite
                score = minimax(board, False, depth=1)

                # Annuler le coup simulé
                board[row][col] = None

                # Conserver le meilleur coup
                if score > best_score:
                    best_score = score
                    best_move = (row, col)

    return best_move
