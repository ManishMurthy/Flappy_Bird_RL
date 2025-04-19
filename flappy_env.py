import pygame
import random
import numpy as np
from collections import deque
import os

class FlappyBirdEnv:
    def __init__(self, width=288, height=512, use_assets=True):
        self.width = width
        self.height = height
        self.pipe_gap = 100  # Gap between pipes
        self.pipe_width = 52
        self.bird_width = 34
        self.bird_height = 24
        self.use_assets = use_assets

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Flappy Bird RL')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Create assets directory if it doesn't exist
        if not os.path.exists('assets'):
            os.makedirs('assets')
            print("Please download and place the following files in the 'assets' directory:")
            print("- background.png (game background)")
            print("- bird1.png, bird2.png, bird3.png (bird animation frames)")
            print("- pipe.png (pipe obstacle)")
            print("- base.png (ground)")
        
        # Load images if assets are available and use_assets is True
        self.bg_color = (135, 206, 250)  # Sky blue (fallback)
        self.bird_color = (255, 255, 0)  # Yellow (fallback)
        self.pipe_color = (0, 128, 0)  # Green (fallback)
        self.base_color = (222, 216, 149)  # Tan (fallback)
        
        if self.use_assets:
            try:
                # Try to load assets
                self.bg_img = self.load_image('assets/background.png')
                self.base_img = self.load_image('assets/base.png')
                self.pipe_img = self.load_image('assets/pipe.png')
                
                # Load bird animation frames
                self.bird_frames = [
                    self.load_image('assets/bird1.png'),
                    self.load_image('assets/bird2.png'),
                    self.load_image('assets/bird3.png')
                ]
                self.bird_frame_idx = 0
                self.animation_time = 0
                
                # Check if all assets loaded successfully
                self.assets_loaded = (self.bg_img and self.base_img and 
                                     self.pipe_img and all(self.bird_frames))
                
                if not self.assets_loaded:
                    print("Some assets failed to load. Using simple graphics instead.")
                else:
                    print("Assets loaded successfully!")
                    
                    # Update dimensions based on actual images
                    if self.bird_frames[0]:
                        self.bird_width, self.bird_height = self.bird_frames[0].get_size()
                    if self.pipe_img:
                        self.pipe_width = self.pipe_img.get_width()
                    
            except Exception as e:
                print(f"Error loading assets: {e}")
                self.assets_loaded = False
        else:
            self.assets_loaded = False
        
        # Game variables
        self.reset()
    
    def load_image(self, path):
        """Load an image and return the surface, or None if it fails"""
        try:
            if os.path.exists(path):
                return pygame.image.load(path).convert_alpha()
            else:
                print(f"Warning: Asset not found: {path}")
                return None
        except pygame.error:
            print(f"Failed to load image: {path}")
            return None
    
    def reset(self):
        """Reset the environment to initial state and return observation"""
        self.bird_x = 50
        self.bird_y = self.height // 2
        self.bird_velocity = 0
        self.gravity = 0.5
        self.flap_power = -8
        
        # Generate first pipes
        self.pipes = []
        self.add_pipe()
        
        self.score = 0
        self.ticks = 0
        self.game_over = False
        
        return self._get_observation()
    
    def add_pipe(self):
        """Add a new pipe to the environment"""
        pipe_y = random.randint(self.pipe_gap + 50, self.height - 50 - self.pipe_gap)
        self.pipes.append({
            'x': self.width,
            'top_height': pipe_y - self.pipe_gap // 2,
            'bottom_y': pipe_y + self.pipe_gap // 2,
            'passed': False
        })
    
    def _get_observation(self):
        """Return the current state observation for the RL agent"""
        # Find the next pipe (the closest one that the bird hasn't passed yet)
        next_pipe = None
        for pipe in self.pipes:
            if pipe['x'] + self.pipe_width > self.bird_x and not pipe['passed']:
                next_pipe = pipe
                break
        
        if next_pipe is None and len(self.pipes) > 0:
            next_pipe = self.pipes[0]
        
        if next_pipe is None:
            # If no pipes, use default values
            horizontal_distance = self.width
            height_diff_top = 0
            height_diff_bottom = 0
        else:
            # Calculate features
            horizontal_distance = next_pipe['x'] - self.bird_x
            height_diff_top = self.bird_y - next_pipe['top_height']
            height_diff_bottom = next_pipe['bottom_y'] - self.bird_y
        
        # Normalize the values
        bird_y_normalized = self.bird_y / self.height
        bird_velocity_normalized = self.bird_velocity / 10.0
        horizontal_distance_normalized = horizontal_distance / self.width
        height_diff_top_normalized = height_diff_top / self.height
        height_diff_bottom_normalized = height_diff_bottom / self.height
        
        return np.array([
            bird_y_normalized,
            bird_velocity_normalized,
            horizontal_distance_normalized,
            height_diff_top_normalized,
            height_diff_bottom_normalized
        ])
    
    def step(self, action):
        """
        Take action in the environment:
        action: 0 = do nothing, 1 = flap
        """
        reward = 0.2  # Small reward for staying alive
        
        # Apply action
        if action == 1:  # Flap
            self.bird_velocity = self.flap_power
        
        # Update bird position
        self.bird_y += self.bird_velocity
        self.bird_velocity += self.gravity
        
        # Update pipes
        for pipe in self.pipes:
            pipe['x'] -= 3  # Pipe speed
            
            # Check if bird has passed the pipe
            if not pipe['passed'] and pipe['x'] + self.pipe_width < self.bird_x:
                pipe['passed'] = True
                self.score += 1
                reward += 2.0  # Reward for passing a pipe
                
        # Remove pipes that have gone off-screen
        self.pipes = [p for p in self.pipes if p['x'] > -self.pipe_width]
        
        # Add new pipes if needed
        if len(self.pipes) < 3:
            if len(self.pipes) == 0 or self.pipes[-1]['x'] < self.width - 150:
                self.add_pipe()
        
        # Check for collisions
        done = self._check_collision()
        if done:
            reward = -1.0  # Penalty for crashing
        
        self.ticks += 1
        
        return self._get_observation(), reward, done, {"score": self.score}
    
    def _check_collision(self):
        """Check if the bird has collided with pipes or boundaries"""
        # Ground position
        base_y = self.height - 70
        
        # Check ceiling and ground boundaries
        if self.bird_y <= 0 or self.bird_y + self.bird_height >= base_y:
            return True
        
        # Check pipe collisions
        for pipe in self.pipes:
            if (pipe['x'] < self.bird_x + self.bird_width and 
                pipe['x'] + self.pipe_width > self.bird_x):
                
                if (self.bird_y < pipe['top_height'] or 
                    self.bird_y + self.bird_height > pipe['bottom_y']):
                    return True
        
        return False
    
    def render(self, mode='human'):
        """Render the current environment state"""
        if self.game_over:
            return
        
        # Update bird animation
        self.animation_time += 1
        if self.animation_time % 5 == 0:  # Change frame every 5 ticks
            self.bird_frame_idx = (self.bird_frame_idx + 1) % 3
            
        # Fill background
        if self.assets_loaded and self.bg_img:
            # Tile the background if needed
            for i in range(0, self.width, self.bg_img.get_width()):
                self.screen.blit(self.bg_img, (i, 0))
        else:
            self.screen.fill(self.bg_color)
        
        # Draw pipes
        for pipe in self.pipes:
            if self.assets_loaded and self.pipe_img:
                # Top pipe (flipped)
                top_pipe = pygame.transform.flip(self.pipe_img, False, True)
                self.screen.blit(top_pipe, (pipe['x'], pipe['top_height'] - top_pipe.get_height()))
                
                # Bottom pipe
                self.screen.blit(self.pipe_img, (pipe['x'], pipe['bottom_y']))
            else:
                # Draw simple rectangles if no assets
                # Top pipe
                pygame.draw.rect(
                    self.screen, 
                    self.pipe_color, 
                    pygame.Rect(pipe['x'], 0, self.pipe_width, pipe['top_height'])
                )
                # Bottom pipe
                pygame.draw.rect(
                    self.screen, 
                    self.pipe_color, 
                    pygame.Rect(pipe['x'], pipe['bottom_y'], self.pipe_width, self.height - pipe['bottom_y'])
                )
        
        # Draw base/ground
        base_y = self.height - 70  # Position of the ground
        if self.assets_loaded and self.base_img:
            for i in range(0, self.width, self.base_img.get_width()):
                self.screen.blit(self.base_img, (i, base_y))
        else:
            pygame.draw.rect(
                self.screen,
                self.base_color,
                pygame.Rect(0, base_y, self.width, 70)
            )
        
        # Draw bird
        if self.assets_loaded and all(self.bird_frames):
            # Get current bird frame
            bird_img = self.bird_frames[self.bird_frame_idx]
            
            # Rotate bird based on velocity
            angle = -self.bird_velocity * 2  # Simple rotation based on velocity
            rotated_bird = pygame.transform.rotate(bird_img, angle)
            
            # Calculate new rect to maintain center position after rotation
            bird_rect = rotated_bird.get_rect(center=(self.bird_x + self.bird_width/2, 
                                                      self.bird_y + self.bird_height/2))
            self.screen.blit(rotated_bird, bird_rect.topleft)
        else:
            # Draw simple rectangle if no assets
            pygame.draw.rect(
                self.screen, 
                self.bird_color, 
                pygame.Rect(self.bird_x, self.bird_y, self.bird_width, self.bird_height)
            )
        
        # Draw score
        score_text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(30)
        
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
                pygame.quit()
    
    def close(self):
        """Close the environment"""
        pygame.quit()