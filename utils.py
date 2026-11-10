import pygame
import numpy as np
import matplotlib.pyplot as plt

class MAZE():
    def __init__(self, config):
        self.MAZE_SIZE = config.MAZE_SIZE
        self.maze = config.maze_setting

        self.Start_point = config.Start_point
        self.End_point = config.End_point

        self.GRID_SIZE = 50
        self.Block_Color = (0, 0, 0)
        self.Line_Color = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.Target_Color = (255, 0, 0)
        self.Destination_color = (255, 255, 0)
        self.WINDOW_WIDTH = self.MAZE_SIZE * self.GRID_SIZE
        self.WINDOW_HEIGHT = self.MAZE_SIZE * self.GRID_SIZE
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))

        self.reset()

    def initialize_maze(self):        
        pygame.init()
        self.screen.fill(self.WHITE)
        
        for i in range(self.MAZE_SIZE + 1):
            pygame.draw.line(self.screen, self.Line_Color, (i * self.GRID_SIZE, 0), (i * self.GRID_SIZE, self.MAZE_SIZE * self.GRID_SIZE))
            pygame.draw.line(self.screen, self.Line_Color, (0, i * self.GRID_SIZE), (self.MAZE_SIZE * self.GRID_SIZE, i * self.GRID_SIZE))
        
        for x in range(self.MAZE_SIZE):
            for y in range(self.MAZE_SIZE):
                if self.maze[x, y] == 1:
                    pygame.draw.rect(self.screen, self.Block_Color, 
                                (x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE))
        
        start_rect = pygame.Rect(self.Start_point[0]*self.GRID_SIZE, self.Start_point[1]*self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.Target_Color, start_rect, width=2)
        
        end_center = (self.End_point[0]*self.GRID_SIZE + self.GRID_SIZE//2, 
                    self.End_point[1]*self.GRID_SIZE + self.GRID_SIZE//2)
        end_radius = self.GRID_SIZE // 2 - 1
        pygame.draw.circle(self.screen, self.Destination_color, end_center, end_radius)
        pygame.display.set_caption("Maze")

    def move_Robot(self, robot_pos):
        self.robot_pos = robot_pos
        self.initialize_maze()
        if robot_pos:
            robot_rect = (robot_pos[0]*self.GRID_SIZE, 
                        robot_pos[1]*self.GRID_SIZE, 
                        self.GRID_SIZE, 
                        self.GRID_SIZE)

        pygame.draw.rect(self.screen, self.Target_Color, robot_rect)
        pygame.display.update()
    
    def close_pygame(self):
        pygame.quit()

    def reset(self):
        self.initialize_maze()
        self.move_Robot(self.Start_point)

    def draw_path(self, path, color=(0, 90, 255)):
        # Draw the visited cells as a line through the cell centres (for screenshots).
        centres = [(p[0]*self.GRID_SIZE + self.GRID_SIZE//2, p[1]*self.GRID_SIZE + self.GRID_SIZE//2) for p in path]
        if len(centres) > 1:
            pygame.draw.lines(self.screen, color, False, centres, 4)
        for c in centres:
            pygame.draw.circle(self.screen, color, c, 5)
        pygame.display.update()

def bfs_shortest_path(maze_setting, start, end):
    # Breadth-first search over free cells; returns the list of cells from start to end.
    size = maze_setting.shape[0]
    start, end = tuple(start), tuple(end)
    parent = {start: None}
    queue = [start]
    for cell in queue:
        if cell == end:
            break
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]: # up, right, down, left
            nxt = (cell[0] + dx, cell[1] + dy)
            if 0 <= nxt[0] < size and 0 <= nxt[1] < size and maze_setting[nxt] == 0 and nxt not in parent:
                parent[nxt] = cell
                queue.append(nxt)
    if end not in parent:
        return None
    path = [end]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return [list(p) for p in reversed(path)]

def draw_rewards(rewards, window = 10, filename = 'Result.jpg', perfect_score = None, title = None):
    smoothed = np.convolve(rewards, np.ones(window), "valid") / window
    plt.figure(figsize=(6,3.2))
    plt.plot(np.arange(1, len(rewards) + 1), rewards, color='0.75', linewidth=0.8, label='Episode reward')
    plt.plot(np.arange(window, len(rewards) + 1), smoothed, linewidth=2, label=f'{window}-episode average')
    if perfect_score is not None:
        plt.axhline(perfect_score, color='green', linestyle='--', label=f'Best possible reward ({perfect_score})')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    if title:
        plt.title(title)
    plt.legend(loc='lower right', fontsize=8)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Results have been saved at '{filename}'.")
