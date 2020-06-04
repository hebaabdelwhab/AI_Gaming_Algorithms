from math import sqrt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from csv import reader
from math import sqrt
import random
import pygame
import random
import sys
import math

# region SearchAlgorithms
class Node:
    id = None  # Unique value for each node.
    up = None  # Represents value of neighbors (up, down, left, right).
    down = None
    left = None
    right = None
    previousNode = None  # Represents value of neighbors.
    def __init__(self, value):
        self.value = value
class SearchAlgorithms:
    i_end = -1
    j_end = -1
    i_start = -1
    j_start = -1
    path = []  # Represents the correct path from start node to the goal node.
    fullPath = []  # Represents all visited nodes from the start node to the goal node.
    def __init__(self, mazeStr):
        self.mzstr = mazeStr
    def unique(self,list1):
        unique_list = []
        for x in list1:
            if x not in unique_list:
                unique_list.append(x)
        return unique_list
    def create_maze(self,str):
        maze = []
        tmp = str.split(' ')
        col = int((len(tmp[0]) + 1) / 2)
        row = len(tmp)
        str = str.replace(',', '')
        str = str.replace(' ', '')
        c = 0
        for i in range(row):
            l = []
            for j in range(col):
                if (str[i] == 'S'):
                    self.start = c
                    self.i_start = i
                    self.j_start = j
                elif(str[i] == 'E'):
                    self.end = c
                    self.i_end = i
                    self.j_end = j
                n = Node(str[c])
                n.id = c
                l.append(n)
                c = c + 1
            maze.append(l)
        startNode = maze[self.i_start][self.j_start]
        Endnode   = maze[self.i_end][self.j_end]
        s = str.index('S')
        e = str.index('E')
        # set adjcent
        for i in range(row):
            for j in range(col):
                tmp = i - 1
                # up
                if (tmp >= 0 and maze[tmp][j].value != '#'):
                    maze[i][j].up = maze[tmp][j].id
                # down
                tmp = i + 1
                if (tmp < row and maze[tmp][j].value != '#'):
                    maze[i][j].down = maze[tmp][j].id
                # left
                tmp = j - 1
                if (tmp >= 0 and maze[i][tmp].value != '#'):
                    maze[i][j].left = maze[i][tmp].id
                # right
                tmp = j + 1
                if (tmp < col and maze[i][tmp].value != '#'):
                    maze[i][j].right = maze[i][tmp].id
        return maze, row, col, s, e, startNode, Endnode
    def dfs(self,string):
        maze, row, col, s, e, startNode, Endnode= self.create_maze(string)
        visited = []
        stack = []
        stack.append(s)
        start = None
        end = None
        while (1 == 1):
            if (len(stack) != 0):
                tmp = stack.pop()
                visited.append(tmp)
                r = int(tmp / col)
                c = (tmp - r * col)

                if (maze[r][c].value == 'E'):
                    start = r
                    end = c
                    break
                if (maze[r][c].right is not None and maze[r][c].right not in visited and maze[r][c].right not in stack):
                    stack.append(maze[r][c].right)
                    maze[r][c+1].previousNode = maze[r][c]
                if (maze[r][c].left is not None and maze[r][c].left not in visited and maze[r][c].left not in stack):
                    stack.append(maze[r][c].left)
                    maze[r][c-1].previousNode = maze[r][c]
                if (maze[r][c].down is not None and maze[r][c].down not in stack and maze[r][c].down not in visited):
                    stack.append(maze[r][c].down)
                    maze[r+1][c].previousNode = maze[r][c]
                if (maze[r][c].up is not None and maze[r][c].up not in stack and maze[r][c].up not in visited):
                    stack.append(maze[r][c].up)
                    maze[r-1][c].previousNode = maze[r][c]

        EndNode = maze[start][end]
        Path = list()
        Path.append(EndNode.id)
        while EndNode.previousNode != None:
            Path.append(EndNode.previousNode.id)
            EndNode = EndNode.previousNode
        Path.reverse()
        visited = self.unique(visited)
        return Path, visited
    def DFS(self):
        self.path, self.fullPath = self.dfs(self.mzstr)
        return  self.fullPath , self.path
# endregion
#region Gaming
class Gaming:
    def __init__(self):
        self.COLOR_BLUE = (0, 0, 240)
        self.COLOR_BLACK = (0, 0, 0)
        self.COLOR_RED = (255, 0, 0)
        self.COLOR_YELLOW = (255, 255, 0)
        self.Y_COUNT = int(5)
        self.X_COUNT = int(8)
        self.PLAYER = 0
        self.AI = 1
        self.PLAYER_PIECE = 1
        self.AI_PIECE = 2
        self.WINNING_WINDOW_LENGTH = 3
        self.EMPTY = 0
        self.WINNING_POSITION = []
        self.SQUARESIZE = 80
        self.width = self.X_COUNT * self.SQUARESIZE
        self.height = (self.Y_COUNT + 1) * self.SQUARESIZE
        self.size = (self.width, self.height)
        self.RADIUS = int(self.SQUARESIZE / 2 - 5)
        self.screen = pygame.display.set_mode(self.size)


    def create_board(self):
        board = np.zeros((self.Y_COUNT, self.X_COUNT))
        return board


    def drop_piece(self, board, row, col, piece):
        board[row][col] = piece


    def is_valid_location(self, board, col):
        return board[self.Y_COUNT - 1][col] == 0


    def get_next_open_row(self, board, col):
        for r in range(self.Y_COUNT):
            if board[r][col] == 0:
                return r


    def print_board(self, board):
        print(np.flip(board, 0))


    def winning_move(self, board, piece):
        self.WINNING_POSITION.clear()
        for c in range(self.X_COUNT - 2):
            for r in range(self.Y_COUNT):
                if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece:
                    self.WINNING_POSITION.append([r, c])
                    self.WINNING_POSITION.append([r, c + 1])
                    self.WINNING_POSITION.append([r, c + 2])
                    return True

        for c in range(self.X_COUNT):
            for r in range(self.Y_COUNT - 2):
                if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece:
                    self.WINNING_POSITION.append([r, c])
                    self.WINNING_POSITION.append([r + 1, c])
                    self.WINNING_POSITION.append([r + 2, c])
                    return True

        for c in range(self.X_COUNT - 2):
            for r in range(self.Y_COUNT - 2):
                if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece:
                    self.WINNING_POSITION.append([r, c])
                    self.WINNING_POSITION.append([r + 1, c + 1])
                    self.WINNING_POSITION.append([r + 2, c + 2])
                    return True

        for c in range(self.X_COUNT - 2):
            for r in range(2, self.Y_COUNT):
                if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece:
                    self.WINNING_POSITION.append([r, c])
                    self.WINNING_POSITION.append([r - 1, c + 1])
                    self.WINNING_POSITION.append([r - 2, c + 2])
                    return True


    def evaluate_window(self, window, piece):
        score = 0
        opp_piece = self.PLAYER_PIECE
        if piece == self.PLAYER_PIECE:
            opp_piece = self.AI_PIECE

        if window.count(piece) == 3:
            score += 100
        elif window.count(piece) == 2 and window.count(self.EMPTY) == 1:
            score += 5

        if window.count(opp_piece) == 3 and window.count(self.EMPTY) == 1:
            score -= 4

        return score


    def score_position(self, board, piece):
        score = 0

        center_array = [int(i) for i in list(board[:, self.X_COUNT // 2])]
        center_count = center_array.count(piece)
        score += center_count * 3

        for r in range(self.Y_COUNT):
            row_array = [int(i) for i in list(board[r, :])]
            for c in range(self.X_COUNT - 3):
                window = row_array[c: c + self.WINNING_WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        for c in range(self.X_COUNT):
            col_array = [int(i) for i in list(board[:, c])]
            for r in range(self.Y_COUNT - 3):
                window = col_array[r: r + self.WINNING_WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        for r in range(self.Y_COUNT - 3):
            for c in range(self.X_COUNT - 3):
                window = [board[r + i][c + i] for i in range(self.WINNING_WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        for r in range(self.Y_COUNT - 3):
            for c in range(self.X_COUNT - 3):
                window = [board[r + 3 - i][c + i] for i in range(self.WINNING_WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        return score


    def is_terminal_node(self, board):
        return self.winning_move(board, self.PLAYER_PIECE) or self.winning_move(board, self.AI_PIECE) or len(
                self.get_valid_locations(board)) == 0


    def AlphaBeta(self, board, depth, alpha, beta, currentPlayer):
        valid_locations = self.get_valid_locations(board)
        value = -math.inf
        column = random.choice(valid_locations)
        if self.is_terminal_node(board) or depth == 0:
            if self.is_terminal_node(board):
                if self.winning_move(board, self.AI_PIECE):
                    return (None, 1000000000000)
                elif self.winning_move(board, self.PLAYER_PIECE):
                    return (None, -1000000000000)
                else:
                    return (None, 0)
            else:
                if currentPlayer:
                    return  None,self.score_position(board,self.PLAYER_PIECE)
                else:
                    return None, self.score_position(board, self.AI_PIECE)
        # initizlation of alpha and byta
        if (currentPlayer):
            value = -math.inf
            for col in valid_locations:
                row = self.get_next_open_row(board, col)
                self.drop_piece(board,row,col,self.AI_PIECE)
                TheScore = self.AlphaBeta(board, depth - 1, alpha, beta, False)[1]
                self.drop_piece(board, row, col, self.EMPTY)
                if TheScore > value:
                    value = TheScore
                    column = col
                alpha = max(alpha,value)
                if beta <= alpha:
                    break
        else:
            value = math.inf
            for col in valid_locations:
                row = self.get_next_open_row(board, col)
                self.drop_piece(board, row, col, self.PLAYER_PIECE)
                TheScore = self.AlphaBeta(board, depth - 1, alpha, beta,True)[1]
                self.drop_piece(board, row, col, self.EMPTY)
                if TheScore < value:
                    value = TheScore
                    column = col
                beta = min(beta,value)
                if beta <= alpha:
                    break
        return column, value
    def get_valid_locations(self, board):
        valid_locations = []
        for col in range(self.X_COUNT):
            if self.is_valid_location(board, col):
                valid_locations.append(col)
        return valid_locations


    def pick_best_move(self, board, piece):
        best_score = -10000
        valid_locations = get_valid_locations(board)
        best_col = random.choice(valid_locations)

        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, piece)
            score = score_position(temp_board, piece)

            if score > best_score:
                best_score = score
                best_col = col

        return best_col


    def draw_board(self, board):
        for c in range(self.X_COUNT):
            for r in range(self.Y_COUNT):
                pygame.draw.rect(self.screen, self.COLOR_BLUE,
                                 (c * self.SQUARESIZE, r * self.SQUARESIZE + self.SQUARESIZE, self.SQUARESIZE,
                                  self.SQUARESIZE))
                pygame.draw.circle(self.screen, self.COLOR_BLACK, (
                        int(c * self.SQUARESIZE + self.SQUARESIZE / 2),
                        int(r * self.SQUARESIZE + self.SQUARESIZE + self.SQUARESIZE / 2)),
                                   self.RADIUS)

        for c in range(self.X_COUNT):
            for r in range(self.Y_COUNT):
                if board[r][c] == self.PLAYER_PIECE:
                    pygame.draw.circle(self.screen, self.COLOR_RED, (
                            int(c * self.SQUARESIZE + self.SQUARESIZE / 2),
                            self.height - int(r * self.SQUARESIZE + self.SQUARESIZE / 2)),
                                       self.RADIUS)
                elif board[r][c] == self.AI_PIECE:
                    pygame.draw.circle(self.screen, self.COLOR_YELLOW, (
                            int(c * self.SQUARESIZE + self.SQUARESIZE / 2),
                            self.height - int(r * self.SQUARESIZE + self.SQUARESIZE / 2)),
                                       self.RADIUS)
        pygame.display.update()
#endregion

# region KMEANS
class DataItem:
    def __init__(self, item):
        self.features = item
        self.clusterId = -1

    def getDataset():
        data = []
        data.append(DataItem([0, 0, 0, 0]))
        data.append(DataItem([0, 0, 0, 1]))
        data.append(DataItem([0, 0, 1, 0]))
        data.append(DataItem([0, 0, 1, 1]))
        data.append(DataItem([0, 1, 0, 0]))
        data.append(DataItem([0, 1, 0, 1]))
        data.append(DataItem([0, 1, 1, 0]))
        data.append(DataItem([0, 1, 1, 1]))
        data.append(DataItem([1, 0, 0, 0]))
        data.append(DataItem([1, 0, 0, 1]))
        data.append(DataItem([1, 0, 1, 0]))
        data.append(DataItem([1, 0, 1, 1]))
        data.append(DataItem([1, 1, 0, 0]))
        data.append(DataItem([1, 1, 0, 1]))
        data.append(DataItem([1, 1, 1, 0]))
        data.append(DataItem([1, 1, 1, 1]))
        return data

class Cluster:
        def __init__(self, id, centroid):
            self.centroid = centroid
            self.data = []
            self.id = id

        def update(self, clusterData):
            self.data = []
            for item in clusterData:
                self.data.append(item.features)
            tmpC = np.average(self.data, axis=0)
            tmpL = []
            for i in tmpC:
                tmpL.append(i)
            self.centroid = tmpL

class SimilarityDistance:
    def euclidean_distance(self, point1, point2):
        result = 0
        for i in range(len(point1)):
            result += (point1[i] - point2[i]) ** 2
        return sqrt(result)

    def Manhattan_distance(self, point1, point2):
        result = 0
        for i in range(len(point1)):
            result += abs((point1[i] - point2[i]))
        return result

class Clustering_kmeans:
        def __init__(self, data, k, noOfIterations, isEuclidean):
            self.data = data
            self.k = k
            self.distance = SimilarityDistance()
            self.noOfIterations = noOfIterations
            self.isEuclidean = isEuclidean

        def initClusters(self):
            self.clusters = []
            for i in range(self.k):
                self.clusters.append(Cluster(i, self.data[i * 10].features))

        def getClusters(self):
            self.initClusters()
            for i in range(self.noOfIterations):
                for item in self.data:
                    MiniDictance = 9999999
                    for cluster in self.clusters:
                        if (self.isEuclidean == 1):
                            ClusterDistance = self.distance.euclidean_distance(cluster.centroid , item.features)
                        else:
                            ClusterDistance = self.distance.Manhattan_distance(cluster.centroid, item.features)
                        if (ClusterDistance <  MiniDictance):
                            item.clusterId = cluster.id
                            MiniDictance = ClusterDistance
                    ClusterData = [j for j in self.data if item.clusterId ==j.clusterId]
                    self.clusters[item.clusterId].update(ClusterData)
            return self.clusters
# endregion
#################################### Algorithms Main Functions #####################################
# region Search_Algorithms_Main_Fn
def SearchAlgorithm_Main():
    searchAlgo = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.')
    fullPath, path = searchAlgo.DFS()
    print('**DFS**\n Full Path is: ' + str(fullPath) +'\n Path is: ' + str(path))

# endregion

#region Gaming_Main_fn
def Gaming_Main():
    game = Gaming()
    board = game.create_board()
    game.print_board(board)
    game_over = False

    pygame.init()

    game.draw_board(board)
    pygame.display.update()

    myfont = pygame.font.SysFont("monospace", 50)
    turn = 1
    while not game_over:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(game.screen, game.COLOR_BLACK, (0, 0, game.width, game.SQUARESIZE))
                posx = event.pos[0]
                if turn == game.PLAYER:
                    pygame.draw.circle(game.screen, game.COLOR_RED, (posx, int(game.SQUARESIZE / 2)), game.RADIUS)

            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(game.screen, game.COLOR_BLACK, (0, 0, game.width, game.SQUARESIZE))

                if turn == game.PLAYER:
                    posx = event.pos[0]
                    col = int(math.floor(posx / game.SQUARESIZE))

                    if game.is_valid_location(board, col):
                        row = game.get_next_open_row(board, col)
                        game.drop_piece(board, row, col, game.PLAYER_PIECE)

                        if game.winning_move(board, game.PLAYER_PIECE):
                            label = myfont.render("Player Human wins!", 1, game.COLOR_RED)
                            print(game.WINNING_POSITION)
                            game.screen.blit(label, (40, 10))
                            game_over = True

                        turn += 1
                        turn = turn % 2

                        # game.print_board(board)
                        game.draw_board(board)

        if turn == game.AI and not game_over:

            col, minimax_score = game.AlphaBeta(board, 5, -math.inf, math.inf, True)

            if game.is_valid_location(board, col):
                row = game.get_next_open_row(board, col)
                game.drop_piece(board, row, col, game.AI_PIECE)

                if game.winning_move(board, game.AI_PIECE):
                    label = myfont.render("Player AI wins!", 1, game.COLOR_YELLOW)
                    print(game.WINNING_POSITION)
                    game.screen.blit(label, (40, 10))
                    game_over = True

                # game.print_board(board)
                game.draw_board(board)

                turn += 1
                turn = turn % 2

        if game_over:
            pygame.time.wait(3000)
            return game.WINNING_POSITION
#endregion


# region KMeans_Main_Fn
def Kmeans_Main():
    dataset = DataItem.getDataset()
    # 1 for Euclidean and 0 for Manhattan
    clustering = Clustering_kmeans(dataset, 2, len(dataset),1)
    clusters = clustering.getClusters()
    for cluster in clusters:
        for i in range(4):
            cluster.centroid[i] = round(cluster.centroid[i], 2)
        print(cluster.centroid[:4])
    return clusters

# endregion


######################## MAIN ###########################33
if __name__ == '__main__':

    SearchAlgorithm_Main()
    Gaming_Main()
    Kmeans_Main()
