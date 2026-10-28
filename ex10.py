```python
import heapq
import random
import pygame

# Configuration
CELL_SIZE = 40
COLS, ROWS = 16, 12
WIDTH = COLS * CELL_SIZE
HEIGHT = ROWS * CELL_SIZE + 60
FPS = 8

DETECTION_RADIUS = 6
ATTACK_RANGE = 1
FLEE_HEALTH_THRESHOLD = 30
PLAYER_DAMAGE = 15
ENEMY_DAMAGE = 10

WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
GRAY = (90, 90, 90)
YELLOW = (255, 213, 79)
BLUE = (66, 135, 245)

STATE_COLORS = {
    "PATROL": (120, 170, 255),
    "CHASE": (255, 152, 0),
    "ATTACK": (244, 67, 54),
    "FLEE": (156, 39, 176),
}


# Grid and walls
def build_walls():
    walls = set()

    for r in range(3, 9):
        walls.add((r, 6))

    for c in range(8, 13):
        walls.add((3, c))

    for r in range(5, 11):
        walls.add((r, 11))

    return walls


def in_bounds(cell):
    r, c = cell
    return 0 <= r < ROWS and 0 <= c < COLS


def neighbors(cell, walls):
    r, c = cell

    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nxt = (r + dr, c + dc)

        if in_bounds(nxt) and nxt not in walls:
            yield nxt


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# A* Search Algorithm
def a_star(start, goal, walls):
    if start == goal:
        return []

    frontier = [(manhattan(start, goal), 0, start)]
    came_from = {}
    g_score = {start: 0}

    while frontier:
        _, g, current = heapq.heappop(frontier)

        if current == goal:
            path = [current]

            while path[-1] != start:
                path.append(came_from[path[-1]])

            path.reverse()
            return path[1:]

        for nxt in neighbors(current, walls):
            new_cost = g + 1

            if nxt not in g_score or new_cost < g_score[nxt]:
                g_score[nxt] = new_cost
                came_from[nxt] = current

                priority = new_cost + manhattan(nxt, goal)
                heapq.heappush(
                    frontier, (priority, new_cost, nxt)
                )

    return []


def farthest_cell_from(target, walls, sample_size=40):
    cells = [
        (r, c)
        for r in range(ROWS)
        for c in range(COLS)
        if (r, c) not in walls
    ]

    candidates = random.sample(
        cells, min(sample_size, len(cells))
    )

    return max(
        candidates,
        key=lambda cell: manhattan(cell, target)
    )


# Enemy AI using Finite State Machine
class EnemyBot:
    def __init__(self, pos, patrol_route, walls):
        self.pos = pos
        self.walls = walls
        self.health = 100
        self.state = "PATROL"
        self.path = []
        self.patrol_route = patrol_route
        self.patrol_index = 0

    def decide_state(self, player_pos):
        distance = manhattan(self.pos, player_pos)

        if self.health < FLEE_HEALTH_THRESHOLD:
            new_state = "FLEE"
        elif distance <= ATTACK_RANGE:
            new_state = "ATTACK"
        elif distance <= DETECTION_RADIUS:
            new_state = "CHASE"
        else:
            new_state = "PATROL"

        if new_state != self.state:
            self.state = new_state
            self.path = []

        return self.state

    def act(self, player_pos):
        if self.state == "ATTACK":
            return ENEMY_DAMAGE

        elif self.state == "CHASE":
            if not self.path or random.random() < 0.3:
                self.path = a_star(
                    self.pos, player_pos, self.walls
                )

            if self.path:
                self.pos = self.path.pop(0)

        elif self.state == "FLEE":
            if not self.path:
                target = farthest_cell_from(
                    player_pos, self.walls
                )
                self.path = a_star(
                    self.pos, target, self.walls
                )

            if self.path:
                self.pos = self.path.pop(0)

        else:
            if not self.path:
                target = self.patrol_route[self.patrol_index]

                if self.pos == target:
                    self.patrol_index = (
                        self.patrol_index + 1
                    ) % len(self.patrol_route)

                    target = self.patrol_route[
                        self.patrol_index
                    ]

                self.path = a_star(
                    self.pos, target, self.walls
                )

            if self.path:
                self.pos = self.path.pop(0)

        return 0


# Drawing functions
def draw_grid(screen, walls):
    for r in range(ROWS):
        for c in range(COLS):
            rect = pygame.Rect(
                c * CELL_SIZE,
                r * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )

            color = WHITE if (r, c) not in walls else GRAY

            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(
                screen, (220, 220, 220), rect, 1
            )


def draw_cell(screen, pos, color, margin=6):
    r, c = pos

    rect = pygame.Rect(
        c * CELL_SIZE + margin,
        r * CELL_SIZE + margin,
        CELL_SIZE - 2 * margin,
        CELL_SIZE - 2 * margin
    )

    pygame.draw.rect(
        screen, color, rect, border_radius=6
    )


def draw_hud(screen, font, player_health, enemy):
    hud_y = ROWS * CELL_SIZE

    pygame.draw.rect(
        screen, BLACK, (0, hud_y, WIDTH, 60)
    )

    text = font.render(
        f"Player HP: {max(player_health, 0)}   "
        f"Enemy HP: {max(enemy.health, 0)}   "
        f"State: {enemy.state}",
        True,
        WHITE
    )

    screen.blit(text, (10, hud_y + 18))


# Main game
def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI Enemy Bot Game")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    walls = build_walls()
    patrol_route = [
        (1, 1), (1, 14), (10, 14), (10, 1)
    ]

    def new_game():
        return {
            "player_pos": (10, 1),
            "player_health": 100,
            "enemy": EnemyBot(
                (1, 14), patrol_route, walls
            ),
            "game_over": False,
            "message": ""
        }

    state = new_game()
    running = True

    while running:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    if state["game_over"]:
                        state = new_game()

                elif not state["game_over"]:

                    r, c = state["player_pos"]
                    new_pos = state["player_pos"]

                    if event.key == pygame.K_UP:
                        new_pos = (r - 1, c)

                    elif event.key == pygame.K_DOWN:
                        new_pos = (r + 1, c)

                    elif event.key == pygame.K_LEFT:
                        new_pos = (r, c - 1)

                    elif event.key == pygame.K_RIGHT:
                        new_pos = (r, c + 1)

                    elif event.key == pygame.K_SPACE:
                        enemy = state["enemy"]

                        if manhattan(
                            state["player_pos"], enemy.pos
                        ) <= ATTACK_RANGE:
                            enemy.health -= PLAYER_DAMAGE

                    if (
                        in_bounds(new_pos)
                        and new_pos not in walls
                    ):
                        state["player_pos"] = new_pos

                    enemy = state["enemy"]

                    enemy.decide_state(state["player_pos"])
                    damage = enemy.act(state["player_pos"])

                    state["player_health"] -= damage

                    if state["player_health"] <= 0:
                        state["game_over"] = True
                        state["message"] = (
                            "You were defeated! Press R to restart."
                        )

                    elif enemy.health <= 0:
                        state["game_over"] = True
                        state["message"] = (
                            "Enemy defeated! Press R to restart."
                        )

        screen.fill(BLACK)

        draw_grid(screen, walls)

        for wall in walls:
            draw_cell(screen, wall, GRAY, margin=0)

        draw_cell(
            screen, state["player_pos"], BLUE
        )

        enemy = state["enemy"]

        draw_cell(
            screen,
            enemy.pos,
            STATE_COLORS[enemy.state]
        )

        draw_hud(
            screen,
            font,
            state["player_health"],
            enemy
        )

        if state["game_over"]:
            msg = font.render(
                state["message"], True, YELLOW
            )

            screen.blit(
                msg, (10, ROWS * CELL_SIZE - 30)
            )

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
```
