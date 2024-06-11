import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

# Constants for the game
SCALING_FACTOR = 10
WIDTH, HEIGHT = 84 * SCALING_FACTOR, 84 * SCALING_FACTOR
NUM_RAYS = 5
TRACK_RADIUS = 30 * SCALING_FACTOR  # Radius of the track
TRACK_WIDTH = 10 * SCALING_FACTOR  # Width of the track, making the drivable area
CENTER = np.array([WIDTH // 2, HEIGHT // 2])
NUM_TRACK_SECTIONS = 360  # Number of sections to divide the track into
STEP_PENALTY = -0.001
D_MAX = TRACK_RADIUS  # Maximum distance for normalization
CHECKPOINTS = [3 * np.pi / 2, np.pi, np.pi / 2]  # Required checkpoints for a lap
LAP_COUNT = 1  # Number of laps required to complete the game
MAX_STEPS_NO_PROGRESS = 240  # Maximum steps allowed without progress

class PolePositionEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 24}

    def __init__(self, render_mode=None, obs_type='features', window_size=(WIDTH, HEIGHT), **kwargs):
        super(PolePositionEnv, self).__init__()

        # Action Space: 5 discrete actions (left, right, accelerate, decelerate, no-op)
        self.action_space = spaces.Discrete(5)

        # Observation Space
        self.obs_type = obs_type

        # Define the observation space based on type
        if self.obs_type == 'pixels':
            self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        elif self.obs_type == 'features':
            low = np.array([0, -1] + [0] * NUM_RAYS, dtype=np.float32)
            high = np.array([1, 1] + [1] * NUM_RAYS, dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.render_mode = render_mode
        self.window_size = window_size

        # Car state variables
        self.car_acceleration = 0.2
        self.car_deceleration = 0.3
        self.car_angle = 0.0  # Heading angle in radians
        self.rotation_speed = 0.05  # radians per frame
        self.car_max_speed = 5.0

        self.reward = 0.0
        self.terminated = False
        self.laps_completed = 0
        self.checkpoints_reached = set()
        self.steps_no_progress = 0

        self.car_image = None
        self.finish_line_image = None

        # Initialize Pygame
        self._setup_pygame()
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reward = 0.0
        self.car_position = np.array(CENTER + [TRACK_RADIUS - TRACK_WIDTH / 2, 0])
        self.car_velocity = np.zeros(2)
        self.car_angle = -np.pi / 2
        self.laps_completed = 0
        self.checkpoints_reached = set()
        self.previous_angle = self.car_angle
        self.track_progress = np.zeros(NUM_TRACK_SECTIONS, dtype=bool)
        self.recent_angles = []  # Track recent angles
        self.steps_no_progress = 0  # Reset the steps without progress counter
        observation = self._get_observation()
        return observation, {}

    def _setup_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
        if self.render_mode == 'human':
            self.car_image = pygame.image.load('PolePosition/envs/resources/car.png').convert_alpha()
            self.car_image = pygame.transform.scale(self.car_image, (4 * SCALING_FACTOR, 2 * SCALING_FACTOR))
            self.finish_line_image = pygame.image.load('PolePosition/envs/resources/finish.png').convert_alpha()
            self.finish_line_image = pygame.transform.scale(self.finish_line_image, (6 * SCALING_FACTOR, 2 * SCALING_FACTOR))
        self.render_surface = pygame.Surface(self.window_size)

    def step(self, action):
        assert self.action_space.contains(action)

        if action == 0:  # Turn left
            self.car_angle += self.rotation_speed
        elif action == 1:  # Turn right
            self.car_angle -= self.rotation_speed
        elif action == 2:  # Accelerate
            self.car_velocity += np.array([math.cos(self.car_angle), math.sin(self.car_angle)]) * self.car_acceleration
        elif action == 3:  # Decelerate
            if np.linalg.norm(self.car_velocity) <= 0.05:
                self.car_velocity = np.zeros(2)  # Stop the car if velocity is very low
            else:
                self.car_velocity *= 1 - self.car_deceleration  # Slow down

        self.car_position += self.car_velocity

        # Clip car velocity to max speed
        if np.linalg.norm(self.car_velocity) > self.car_max_speed:
            self.car_velocity = self.car_velocity / np.linalg.norm(self.car_velocity) * self.car_max_speed

        # Calculate reward and check if terminated
        reward, self.terminated = self._calculate_reward()

        current_angle = math.atan2(*(self.car_position - CENTER)[::-1])
        self._update_track_progress(current_angle)

        observation = self._get_observation()
        truncated = False

        return observation, reward, bool(self.terminated), bool(truncated), {}

    def _calculate_ray_intersection(self, angle):
        for d in range(1000):
            end_x = self.car_position[0] + d * math.cos(angle)
            end_y = self.car_position[1] + d * math.sin(angle)
            point = np.array([end_x, end_y])
            distance_from_center = np.linalg.norm(point - CENTER)
            if distance_from_center < TRACK_RADIUS - TRACK_WIDTH or distance_from_center > TRACK_RADIUS:
                return point
        return self.car_position + np.array([1000 * math.cos(angle), 1000 * math.sin(angle)])

    def _draw_rays(self):
        angles = [self.car_angle + np.deg2rad(x) for x in [-90, -45, 0, 45, 90]]
        for angle in angles:
            end_point = self._calculate_ray_intersection(angle)
            pygame.draw.line(self.render_surface, (0, 0, 255), self.car_position, end_point, 2)

    def _draw_track(self):
        for i in range(NUM_TRACK_SECTIONS):
            angle = i * 2 * np.pi / NUM_TRACK_SECTIONS
            next_angle = (i + 1) * 2 * np.pi / NUM_TRACK_SECTIONS
            start_outer = CENTER + (TRACK_RADIUS * np.array([math.cos(angle), math.sin(angle)]))
            end_outer = CENTER + (TRACK_RADIUS * np.array([math.cos(next_angle), math.sin(next_angle)]))
            start_inner = CENTER + ((TRACK_RADIUS - TRACK_WIDTH) * np.array([math.cos(angle), math.sin(angle)]))
            end_inner = CENTER + ((TRACK_RADIUS - TRACK_WIDTH) * np.array([math.cos(next_angle), math.sin(next_angle)]))
            color = (0, 255, 0) if self.track_progress[i] else (200, 200, 200)
            pygame.draw.polygon(self.render_surface, color, [start_outer, end_outer, end_inner, start_inner])

    def _draw_finish_line(self):
        if self.obs_type == 'pixels':
            finish_line_image = pygame.image.load('PolePosition/envs/resources/finish.png').convert_alpha()
            finish_line_image = pygame.transform.scale(finish_line_image, (6 * SCALING_FACTOR, 2 * SCALING_FACTOR))
            finish_line_rect = finish_line_image.get_rect(center=CENTER + [TRACK_RADIUS - TRACK_WIDTH / 2, 0])
            self.render_surface.blit(finish_line_image, finish_line_rect.topleft)
        else:
            finish_line_rect = pygame.Rect(CENTER + [TRACK_RADIUS - TRACK_WIDTH / 2, 0] - [3 * SCALING_FACTOR, 1 * SCALING_FACTOR], (6 * SCALING_FACTOR, 2 * SCALING_FACTOR))
            pygame.draw.rect(self.render_surface, (255, 255, 255), finish_line_rect)

    def _draw_car(self):
        if self.obs_type == 'pixels':
            if self.car_image is None:
                self.car_image = pygame.image.load('PolePosition/envs/resources/car.png').convert_alpha()
                self.car_image = pygame.transform.scale(self.car_image, (4 * SCALING_FACTOR, 2 * SCALING_FACTOR))
            rotated_car = pygame.transform.rotate(self.car_image, -math.degrees(self.car_angle))
            car_rect = rotated_car.get_rect(center=self.car_position)
            self.render_surface.blit(rotated_car, car_rect.topleft)
        else:
            car_surface = pygame.Surface((4 * SCALING_FACTOR, 2 * SCALING_FACTOR), pygame.SRCALPHA)
            car_surface.fill((255, 0, 0))
            rotated_car = pygame.transform.rotate(car_surface, -math.degrees(self.car_angle))
            car_rect = rotated_car.get_rect(center=self.car_position)
            self.render_surface.blit(rotated_car, car_rect.topleft)

    def _update_track_progress(self, current_angle):
        section_index = int((current_angle % (2 * np.pi)) / (2 * np.pi) * NUM_TRACK_SECTIONS)
        if not self.track_progress[section_index]:
            self.track_progress[section_index] = True
            self.steps_no_progress = 0  # Reset the steps without progress counter if progress is made
        else:
            self.steps_no_progress += 1

    def _handle_pygame_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()

    def render(self):
        self._handle_pygame_events()
        if self.render_mode == 'rgb_array':
            return self._render_frame(mode='rgb_array')
        elif self.render_mode == 'human':
            self._render_frame(mode='human')
            return None

    def _render_frame(self, mode='human'):
        self.render_surface.fill((0, 0, 0))
        self._draw_track()
        self._draw_finish_line()
        self._draw_car()
        self._draw_rays()

        if mode == 'human':
            # Scale the render_surface to the current window size
            window_size = self.screen.get_size()
            scaled_surface = pygame.transform.scale(self.render_surface, window_size)
            self.screen.blit(scaled_surface, (0, 0))
            pygame.display.flip()
        elif mode == 'rgb_array':
            # Return the rendered frame as an RGB array
            pixels = pygame.surfarray.array3d(self.render_surface)
            pixels = np.transpose(pixels, (1, 0, 2))
            resized_pixels = pygame.transform.smoothscale(self.render_surface, (84, 84))
            return pygame.surfarray.array3d(resized_pixels).transpose(1, 0, 2)

    def close(self):
        pygame.quit()

    def _get_observation(self):
        if self.obs_type == 'features':
            speed = np.linalg.norm(self.car_velocity)
            normalized_velocity = speed / self.car_max_speed
            normalized_angle = self.car_angle / np.pi
            proximities = [np.linalg.norm(self._calculate_ray_intersection(self.car_angle + np.deg2rad(x)) - self.car_position) for x in [-90, -45, 0, 45, 90]]
            normalized_proximities = [min(p, D_MAX) / D_MAX for p in proximities]
            return np.array([normalized_velocity, normalized_angle] + normalized_proximities, dtype=np.float32)
        elif self.obs_type == 'pixels':
            pixels = self.render()
            if pixels is not None:
                return pixels
            else:
                return np.zeros((84, 84, 3), dtype=np.uint8)

    def _calculate_reward(self):
        self.reward = STEP_PENALTY  # Small penalty for each step to encourage efficient movement
        self.terminated = False

        current_angle = math.atan2(*(self.car_position - CENTER)[::-1])
        normalized_angle = (current_angle + 2 * np.pi) % (2 * np.pi)
        angle_diff = (normalized_angle - self.previous_angle)

        # Reward for progress
        if angle_diff < 0:
            progress_reward = abs(angle_diff) / (2 * np.pi)
            self.reward += (progress_reward * 2)

        self.previous_angle = normalized_angle

        for checkpoint in CHECKPOINTS:
            if abs(normalized_angle - checkpoint) < 0.1:
                self.checkpoints_reached.add(checkpoint)

        if len(self.checkpoints_reached) == len(CHECKPOINTS) and (abs(normalized_angle) < 0.1):
            self.laps_completed += 1
            self.checkpoints_reached.clear()
            self.reward += 5.0  # Increased reward for completing a lap
            if self.laps_completed >= LAP_COUNT:
                self.terminated = True

        distance_from_center = np.linalg.norm(self.car_position - CENTER)
        if not (TRACK_RADIUS - TRACK_WIDTH <= distance_from_center <= TRACK_RADIUS):
            self.reward -= 1.0
            self.terminated = True

        # Terminate if no progress is made for too long
        if self.steps_no_progress > MAX_STEPS_NO_PROGRESS:
            self.terminated = True
            self.reward -= 10.0

        return self.reward, self.terminated
