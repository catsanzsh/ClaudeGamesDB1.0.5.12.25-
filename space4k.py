import pygame
import numpy as np
import sounddevice as sd
import threading
import time
import queue
import random
import math

# --- Pygame Configuration ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (60, 60, 255)  # Player color
YELLOW = (255, 255, 0)  # Player bullet
CYAN = (0, 255, 255)  # Alien bullet
ORANGE = (255, 165, 0)  # Alien color 1
PINK = (255, 105, 180)  # Alien color 2
PURPLE = (160, 32, 240)  # Alien color 3
SHIELD_COLOR = (40, 180, 40)  # Shield green
FPS = 60

# --- Game Settings ---
PLAYER_SPEED = 5
PLAYER_LIVES = 3
PLAYER_BULLET_SPEED = 7  # Positive: Moving UP
PLAYER_MAX_BULLETS = 1  # Classic Invaders limit
ALIEN_BULLET_SPEED = 5  # Positive magnitude; direction handled in Bullet
ALIEN_SHOOT_CHANCE = 0.002  # Chance per alien per frame to shoot
SHIELD_BLOCK_SIZE = 8
INITIAL_ALIEN_MOVE_INTERVAL = 50  # Frames between moves (lower is faster)

# --- Alien Grid & Behavior Constants ---
ALIEN_ROWS = 5
ALIEN_COLS = 11
ALIEN_MOVE_DOWN_STEP = 8
ALIEN_NOTES = ["C2", "D2", "E2", "F2"]

# --- Sound Engine ---
SAMPLE_RATE = 44100
BUFFER_SIZE = 512
MASTER_VOLUME = 0.1
_sound_thread = None
_stop_flag = threading.Event()
_command_queue = queue.Queue(maxsize=100)
_sound_engine_active = False

NOTE_NAMES = ["C", "CS", "D", "DS", "E", "F", "FS", "G", "GS", "A", "AS", "B"]
BASE_FREQ_A4 = 440.0
NOTE_A4_INDEX = NOTE_NAMES.index("A")
OCTAVE_A4 = 4
SEMITONES_A4_FROM_C0 = OCTAVE_A4 * 12 + NOTE_A4_INDEX

# --- Sound Engine Functions ---

def note_to_freq(note_name):
    """Convert a note name like 'A4' to its frequency in Hz"""
    if not note_name:
        return 0
    
    # Handle 'REST' as a special case
    if note_name.upper() == "REST":
        return 0
    
    # Parse the note name and octave
    if note_name[-1].isdigit():
        note = note_name[:-1].upper()
        octave = int(note_name[-1])
    else:
        note = note_name.upper()
        octave = 4  # Default to octave 4 if not specified
    
    # Handle sharps
    if note.endswith("S"):
        note_without_s = note[:-1]
        if note_without_s in NOTE_NAMES:
            semi_from_c0 = octave * 12 + NOTE_NAMES.index(note)
        else:
            raise ValueError(f"Invalid note name: {note_name}")
    else:
        if note in NOTE_NAMES:
            semi_from_c0 = octave * 12 + NOTE_NAMES.index(note)
        else:
            raise ValueError(f"Invalid note name: {note_name}")
            
    # Calculate semitones from A4
    semi_from_a4 = semi_from_c0 - SEMITONES_A4_FROM_C0
    
    # Convert to frequency using equal temperament formula:
    # f = f_A4 * 2^(n/12)
    freq = BASE_FREQ_A4 * (2 ** (semi_from_a4 / 12.0))
    
    return freq

def sine_wave(freq, duration, volume=0.5, phase=0):
    """Generate a sine wave at the given frequency, duration, and volume"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.sin(2 * np.pi * freq * t + phase) * volume
    return wave

def square_wave(freq, duration, volume=0.5, duty_cycle=0.5):
    """Generate a square wave with the given frequency, duration, volume, and duty cycle"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.where((np.sin(2 * np.pi * freq * t) > 0) == (np.random.random() < duty_cycle),
                   volume, -volume)
    return wave

def triangle_wave(freq, duration, volume=0.5):
    """Generate a triangle wave with the given frequency, duration, and volume"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = 2 * volume * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - volume
    return wave

def sawtooth_wave(freq, duration, volume=0.5):
    """Generate a sawtooth wave with the given frequency, duration, and volume"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = 2 * volume * (t * freq - np.floor(t * freq + 0.5))
    return wave

def noise(duration, volume=0.5):
    """Generate white noise with the given duration and volume"""
    return np.random.uniform(-volume, volume, int(SAMPLE_RATE * duration))

def simple_envelope(wave, attack=0.01, decay=0.1, sustain=0.7, release=0.2):
    """Apply a simple ADSR envelope to a wave"""
    samples = len(wave)
    envelope = np.ones(samples)
    
    # Calculate how many samples for each stage
    attack_samples = int(attack * samples)
    decay_samples = int(decay * samples)
    release_samples = int(release * samples)
    sustain_samples = samples - attack_samples - decay_samples - release_samples
    
    # Create the envelope
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    if decay_samples > 0:
        envelope[attack_samples:attack_samples+decay_samples] = np.linspace(1, sustain, decay_samples)
    if sustain_samples > 0:
        envelope[attack_samples+decay_samples:attack_samples+decay_samples+sustain_samples] = sustain
    if release_samples > 0:
        envelope[attack_samples+decay_samples+sustain_samples:] = np.linspace(sustain, 0, release_samples)
    
    return wave * envelope

def sound_thread_func():
    """The main function for the sound thread"""
    global _sound_engine_active
    
    def audio_callback(outdata, frames, time, status):
        # Just fill with zeros (silence) if nothing to play
        if _command_queue.empty():
            outdata.fill(0)
            return
        
        # Get next command and generate audio data
        command = _command_queue.get()
        wave_type = command.get("wave_type", "sine")
        freq = command.get("freq", 440)
        duration = command.get("duration", 0.1)
        volume = command.get("volume", 0.5) * MASTER_VOLUME
        
        # Generate the appropriate wave
        if freq == 0:  # For REST
            wave = np.zeros(frames)
        elif wave_type == "sine":
            wave = sine_wave(freq, duration=frames/SAMPLE_RATE, volume=volume)
        elif wave_type == "square":
            wave = square_wave(freq, duration=frames/SAMPLE_RATE, volume=volume)
        elif wave_type == "triangle":
            wave = triangle_wave(freq, duration=frames/SAMPLE_RATE, volume=volume)
        elif wave_type == "sawtooth":
            wave = sawtooth_wave(freq, duration=frames/SAMPLE_RATE, volume=volume)
        elif wave_type == "noise":
            wave = noise(duration=frames/SAMPLE_RATE, volume=volume)
        else:
            wave = np.zeros(frames)
        
        # Apply envelope if specified
        if "envelope" in command:
            env = command["envelope"]
            wave = simple_envelope(wave, 
                                  attack=env.get("attack", 0.01),
                                  decay=env.get("decay", 0.1), 
                                  sustain=env.get("sustain", 0.7),
                                  release=env.get("release", 0.2))
        
        # Make sure the wave has the right shape for the output buffer
        wave = np.resize(wave, (frames, 1))
        outdata[:] = wave
    
    # Start the audio stream
    stream = sd.OutputStream(samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE,
                            channels=1, callback=audio_callback)
    stream.start()
    _sound_engine_active = True
    
    # Run until stop flag is set
    while not _stop_flag.is_set():
        time.sleep(0.01)  # Just to avoid consuming CPU
    
    # Clean up
    stream.stop()
    stream.close()
    _sound_engine_active = False

def start_sound_engine():
    """Start the sound engine thread if it's not already running"""
    global _sound_thread, _stop_flag
    
    if _sound_thread is None or not _sound_thread.is_alive():
        _stop_flag.clear()
        _sound_thread = threading.Thread(target=sound_thread_func)
        _sound_thread.daemon = True
        _sound_thread.start()

def stop_sound_engine():
    """Stop the sound engine thread"""
    global _sound_thread, _stop_flag
    
    if _sound_thread is not None and _sound_thread.is_alive():
        _stop_flag.set()
        _sound_thread.join(timeout=1.0)
        _sound_thread = None

def play_sound(wave_type="sine", note=None, freq=None, duration=0.1, volume=0.5, 
              attack=0.01, decay=0.1, sustain=0.7, release=0.2):
    """Queue a sound to play"""
    # Convert note to frequency if specified
    if note is not None and freq is None:
        freq = note_to_freq(note)
    elif freq is None:
        freq = 440  # Default to A4
    
    # Create the command dict
    command = {
        "wave_type": wave_type,
        "freq": freq,
        "duration": duration,
        "volume": volume,
        "envelope": {
            "attack": attack,
            "decay": decay,
            "sustain": sustain,
            "release": release
        }
    }
    
    # Make sure the sound engine is running
    if not _sound_engine_active:
        start_sound_engine()
    
    # Queue the command
    try:
        _command_queue.put_nowait(command)
    except queue.Full:
        pass  # Just discard if queue is full

# --- Game Classes ---

class Player(pygame.sprite.Sprite):
    """The player's ship at the bottom of the screen"""
    def __init__(self):
        super().__init__()
        
        # Create player ship
        self.width = 40
        self.height = 30
        self.image = pygame.Surface((self.width, self.height))
        self.image.fill(BLACK)
        
        # Draw a triangular ship
        pygame.draw.polygon(self.image, BLUE, [
            (0, self.height),
            (self.width // 2, 0),
            (self.width, self.height)
        ])
        
        self.rect = self.image.get_rect()
        self.rect.centerx = SCREEN_WIDTH // 2
        self.rect.bottom = SCREEN_HEIGHT - 20
        
        self.speed = PLAYER_SPEED
        self.lives = PLAYER_LIVES
        self.score = 0
        
    def update(self, keys=None):
        """Update the player's position based on key presses"""
        if keys is None:
            return
            
        if keys[pygame.K_LEFT] and self.rect.left > 0:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT] and self.rect.right < SCREEN_WIDTH:
            self.rect.x += self.speed
            
    def shoot(self):
        """Create a new bullet and play a sound"""
        bullet = Bullet(self.rect.centerx, self.rect.top, -PLAYER_BULLET_SPEED)
        play_sound(wave_type="triangle", note="C5", duration=0.1, volume=0.3)
        return bullet
    
    def hit(self):
        """Handle when the player is hit by an alien bullet"""
        self.lives -= 1
        play_sound(wave_type="noise", duration=0.5, volume=0.4)
        return self.lives <= 0

class Alien(pygame.sprite.Sprite):
    """An alien enemy"""
    def __init__(self, row, col):
        super().__init__()
        
        self.row = row
        self.col = col
        
        # Different colors/shapes based on row
        colors = [ORANGE, PINK, PURPLE, GREEN, RED]
        self.color = colors[min(row, len(colors) - 1)]
        
        # Create alien sprite
        self.width = 30
        self.height = 30
        self.image = pygame.Surface((self.width, self.height))
        self.image.fill(BLACK)
        
        # Draw the alien shape
        pygame.draw.ellipse(self.image, self.color, [0, 0, self.width, self.height])
        
        # Add eyes
        pygame.draw.circle(self.image, WHITE, (self.width // 3, self.height // 3), 3)
        pygame.draw.circle(self.image, WHITE, (2 * self.width // 3, self.height // 3), 3)
        
        self.rect = self.image.get_rect()
        
        # Position based on grid coordinates
        self.rect.x = 50 + col * (self.width + 20)
        self.rect.y = 50 + row * (self.height + 15)
        
        # Points value based on row (higher rows worth more)
        self.points = (ALIEN_ROWS - row) * 10
        
    def update(self, dx=0, dy=0):
        """Move the alien by the given amount"""
        self.rect.x += dx
        self.rect.y += dy
        
    def shoot(self):
        """Create a bullet shot by this alien"""
        bullet = Bullet(self.rect.centerx, self.rect.bottom, ALIEN_BULLET_SPEED)
        return bullet

class Bullet(pygame.sprite.Sprite):
    """A bullet fired by the player or aliens"""
    def __init__(self, x, y, speed):
        super().__init__()
        
        self.width = 4
        self.height = 10
        self.image = pygame.Surface((self.width, self.height))
        
        # Color based on who fired it
        if speed < 0:  # Player bullet (moving up)
            self.image.fill(YELLOW)
        else:  # Alien bullet (moving down)
            self.image.fill(CYAN)
        
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.y = y
        
        self.speed = speed  # Negative for up, positive for down
        
    def update(self):
        """Move the bullet"""
        self.rect.y += self.speed
        
        # Destroy if out of bounds
        if self.rect.bottom < 0 or self.rect.top > SCREEN_HEIGHT:
            self.kill()

class Shield(pygame.sprite.Sprite):
    """A destructible shield to protect the player"""
    def __init__(self, x, y):
        super().__init__()
        
        self.width = 60
        self.height = 50
        self.image = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        
        # Create shield grid (to track which blocks are still active)
        self.blocks = {}
        
        # Draw the shield shape
        for bx in range(0, self.width, SHIELD_BLOCK_SIZE):
            for by in range(0, self.height, SHIELD_BLOCK_SIZE):
                # Skip corners to create an arch shape
                if (bx < SHIELD_BLOCK_SIZE and by < SHIELD_BLOCK_SIZE) or \
                   (bx > self.width - SHIELD_BLOCK_SIZE*2 and by < SHIELD_BLOCK_SIZE) or \
                   (bx < SHIELD_BLOCK_SIZE and by > self.height - SHIELD_BLOCK_SIZE*2) or \
                   (bx > self.width - SHIELD_BLOCK_SIZE*2 and by > self.height - SHIELD_BLOCK_SIZE*2):
                    continue
                
                # Skip the bottom center to create an entrance
                if by > self.height - SHIELD_BLOCK_SIZE and \
                   bx > self.width//3 and bx < 2*self.width//3:
                    continue
                
                # Add this block and draw it
                block_key = (bx, by)
                self.blocks[block_key] = True
                pygame.draw.rect(self.image, SHIELD_COLOR, 
                                [bx, by, SHIELD_BLOCK_SIZE, SHIELD_BLOCK_SIZE])
    
    def check_collision(self, bullet):
        """Check if a bullet hit this shield, and damage it if so"""
        if not pygame.sprite.collide_rect(self, bullet):
            return False
            
        # Get the relative position of the bullet in the shield
        rel_x = bullet.rect.x - self.rect.x
        rel_y = bullet.rect.y - self.rect.y
        
        # Round to the nearest block
        block_x = (rel_x // SHIELD_BLOCK_SIZE) * SHIELD_BLOCK_SIZE
        block_y = (rel_y // SHIELD_BLOCK_SIZE) * SHIELD_BLOCK_SIZE
        
        # Check if this block exists and can be damaged
        block_key = (block_x, block_y)
        if block_key in self.blocks and self.blocks[block_key]:
            # Damage the block
            self.blocks[block_key] = False
            
            # Redraw the shield
            self.redraw()
            
            # Remove the bullet
            bullet.kill()
            return True
            
        return False
            
    def redraw(self):
        """Redraw the shield based on the current block state"""
        # Clear the image
        self.image.fill((0, 0, 0, 0))
        
        # Redraw the active blocks
        for (bx, by), active in self.blocks.items():
            if active:
                pygame.draw.rect(self.image, SHIELD_COLOR, 
                                [bx, by, SHIELD_BLOCK_SIZE, SHIELD_BLOCK_SIZE])

class Game:
    """Main game class to manage the Space Invaders game"""
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Space Invaders")
        
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24)
        self.game_over_font = pygame.font.SysFont('Arial', 48)
        
        self.player = None
        self.aliens = pygame.sprite.Group()
        self.player_bullets = pygame.sprite.Group()
        self.alien_bullets = pygame.sprite.Group()
        self.shields = pygame.sprite.Group()
        
        self.alien_move_counter = 0
        self.alien_move_interval = INITIAL_ALIEN_MOVE_INTERVAL
        self.alien_move_right = True
        self.alien_move_down = False
        
        self.game_over = False
        self.win = False
        self.level = 1
        
        # Start the sound engine
        start_sound_engine()
        
    def setup_level(self):
        """Set up a new level"""
        # Clear all groups
        self.aliens.empty()
        self.player_bullets.empty()
        self.alien_bullets.empty()
        self.shields.empty()
        
        # Create player if needed
        if self.player is None:
            self.player = Player()
        
        # Create the alien grid
        for row in range(ALIEN_ROWS):
            for col in range(ALIEN_COLS):
                alien = Alien(row, col)
                self.aliens.add(alien)
        
        # Create shields
        for i in range(4):
            shield = Shield(150 + i * 170, SCREEN_HEIGHT - 150)
            self.shields.add(shield)
        
        # Reset movement
        self.alien_move_counter = 0
        self.alien_move_interval = max(INITIAL_ALIEN_MOVE_INTERVAL - (self.level * 5), 10)
        self.alien_move_right = True
        self.alien_move_down = False
        
        # Reset state
        self.game_over = False
        self.win = False
        
    def handle_events(self):
        """Handle game events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE and not self.game_over:
                    self.player_shoot()
                elif event.key == pygame.K_r and self.game_over:
                    self.reset_game()
                elif event.key == pygame.K_n and self.win:
                    self.next_level()
        
        return True
        
    def player_shoot(self):
        """Handle player shooting"""
        # Check if maximum bullets reached
        if len(self.player_bullets) < PLAYER_MAX_BULLETS:
            bullet = self.player.shoot()
            self.player_bullets.add(bullet)
    
    def alien_shoot(self):
        """Handle alien shooting"""
        # Find aliens in the bottom row of each column
        bottom_aliens = {}
        for alien in self.aliens:
            if alien.col not in bottom_aliens or alien.rect.bottom > bottom_aliens[alien.col].rect.bottom:
                bottom_aliens[alien.col] = alien
        
        # Chance for each bottom alien to shoot
        for alien in bottom_aliens.values():
            if random.random() < ALIEN_SHOOT_CHANCE:
                bullet = alien.shoot()
                self.alien_bullets.add(bullet)
                play_sound(wave_type="square", note="E3", duration=0.1, volume=0.2)
    
    def move_aliens(self):
        """Move all aliens according to the current direction"""
        # Check if it's time to move
        self.alien_move_counter += 1
        if self.alien_move_counter < self.alien_move_interval:
            return
            
        self.alien_move_counter = 0
        
        # Find leftmost and rightmost aliens
        left_edge = SCREEN_WIDTH
        right_edge = 0
        for alien in self.aliens:
            if alien.rect.left < left_edge:
                left_edge = alien.rect.left
            if alien.rect.right > right_edge:
                right_edge = alien.rect.right
        
        # Check if we need to change direction
        if self.alien_move_down:
            # After moving down, switch horizontal direction
            self.alien_move_down = False
            self.alien_move_right = not self.alien_move_right
            
            # Move all aliens down
            for alien in self.aliens:
                alien.update(0, ALIEN_MOVE_DOWN_STEP)
                
            # Play sound
            play_sound(wave_type="sawtooth", note=random.choice(ALIEN_NOTES), 
                      duration=0.2, volume=0.3)
        else:
            # Check if we hit an edge
            if self.alien_move_right and right_edge >= SCREEN_WIDTH - 20:
                self.alien_move_down = True
                return
            elif not self.alien_move_right and left_edge <= 20:
                self.alien_move_down = True
                return
            
            # Move horizontally
            dx = 5 if self.alien_move_right else -5
            for alien in self.aliens:
                alien.update(dx, 0)
                
            # Play move sound
            if len(self.aliens) > 0:
                play_sound(wave_type="square", note=random.choice(ALIEN_NOTES), 
                          duration=0.1, volume=0.2)
    
    def check_collisions(self):
        """Check for all game collisions"""
        # Player bullets hitting aliens
        hits = pygame.sprite.groupcollide(self.player_bullets, self.aliens, True, True)
        for bullet, hit_aliens in hits.items():
            for alien in hit_aliens:
                self.player.score += alien.points
                play_sound(wave_type="sawtooth", note="A4", duration=0.1, volume=0.3)
        
        # Alien bullets hitting player
        if not self.game_over:
            hits = pygame.sprite.spritecollide(self.player, self.alien_bullets, True)
            if hits:
                self.game_over = self.player.hit()
        
        # Bullets hitting shields
        for shield in self.shields:
            # Check player bullets
            for bullet in list(self.player_bullets):
                if shield.check_collision(bullet):
                    play_sound(wave_type="noise", duration=0.05, volume=0.1)
            
            # Check alien bullets
            for bullet in list(self.alien_bullets):
                if shield.check_collision(bullet):
                    play_sound(wave_type="noise", duration=0.05, volume=0.1)
        
        # Check if aliens reached the bottom
        for alien in self.aliens:
            if alien.rect.bottom >= self.player.rect.top:
                self.game_over = True
                break
        
        # Check if all aliens are dead
        if len(self.aliens) == 0:
            self.win = True
    
    def update(self):
        """Update the game state"""
        if self.game_over or self.win:
            return
            
        # Get keyboard state
        keys = pygame.key.get_pressed()
        
        # Update player
        self.player.update(keys)
        
        # Update all bullets
        self.player_bullets.update()
        self.alien_bullets.update()
        
        # Move aliens
        self.move_aliens()
        
        # Aliens shoot
        self.alien_shoot()
        
        # Check collisions
        self.check_collisions()
    
    def draw(self):
        """Draw the game"""
        # Clear screen
        self.screen.fill(BLACK)
        
        # Draw player, aliens, bullets, and shields
        if self.player:
            self.screen.blit(self.player.image, self.player.rect)
        
        self.aliens.draw(self.screen)
        self.player_bullets.draw(self.screen)
        self.alien_bullets.draw(self.screen)
        self.shields.draw(self.screen)
        
        # Draw score and lives
        score_text = self.font.render(f"Score: {self.player.score}", True, WHITE)
        lives_text = self.font.render(f"Lives: {self.player.lives}", True, WHITE)
        level_text = self.font.render(f"Level: {self.level}", True, WHITE)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lives_text, (10, 40))
        self.screen.blit(level_text, (SCREEN_WIDTH - 100, 10))
        
        # Draw game over or win message
        if self.game_over:
            game_over_text = self.game_over_font.render("GAME OVER", True, RED)
            restart_text = self.font.render("Press 'R' to Restart", True, WHITE)
            
            self.screen.blit(game_over_text, (SCREEN_WIDTH//2 - game_over_text.get_width()//2, 
                                            SCREEN_HEIGHT//2 - game_over_text.get_height()//2))
            self.screen.blit(restart_text, (SCREEN_WIDTH//2 - restart_text.get_width()//2, 
                                          SCREEN_HEIGHT//2 + 50))
        elif self.win:
            win_text = self.game_over_font.render("LEVEL COMPLETE!", True, GREEN)
            next_text = self.font.render("Press 'N' for Next Level", True, WHITE)
            
            self.screen.blit(win_text, (SCREEN_WIDTH//2 - win_text.get_width()//2, 
                                      SCREEN_HEIGHT//2 - win_text.get_height()//2))
            self.screen.blit(next_text, (SCREEN_WIDTH//2 - next_text.get_width()//2, 
                                       SCREEN_HEIGHT//2 + 50))
        
        # Update display
        pygame.display.flip()
    
    def reset_game(self):
        """Reset the game after game over"""
        self.player = None
        self.level = 1
        self.setup_level()
    
    def next_level(self):
        """Advance to the next level"""
        self.level += 1
        self.setup_level()
    
    def run(self):
        """Main game loop"""
        # Set up the initial level
        self.setup_level()
        
        running = True
        while running:
            # Handle events
            running = self.handle_events()
            
            # Update game state
            self.update()
            
            # Draw the game
            self.draw()
            
            # Cap the frame rate
            self.clock.tick(FPS)
        
        # Clean up
        stop_sound_engine()
        pygame.quit()

# --- Main Execution ---
if __name__ == "__main__":
    game = Game()
    game.run()
