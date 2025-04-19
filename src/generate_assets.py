from PIL import Image, ImageDraw
import random
import os

def generate_assets():
    """Generate simple placeholder assets for the Flappy Bird game"""
    # Create assets directory if it doesn't exist
    if not os.path.exists('assets'):
        os.makedirs('assets')
    
    print("Generating placeholder assets...")
    
    # Create a simple bird
    for i in range(1, 4):
        bird = Image.new('RGBA', (34, 24), (0, 0, 0, 0))
        draw = ImageDraw.Draw(bird)
        
        # Main body
        draw.ellipse((2, 2, 32, 22), fill=(255, 255, 0))
        
        # Eye
        draw.ellipse((25, 8 + i, 32, 15 + i), fill=(255, 255, 255))
        draw.ellipse((27, 10 + i, 30, 13 + i), fill=(0, 0, 0))
        
        # Beak
        draw.polygon([(32, 15), (40, 13), (32, 18)], fill=(255, 165, 0))
        
        # Wing position varies slightly for animation
        wing_y = 12 + 2 * (i - 1)
        draw.ellipse((10, wing_y, 25, wing_y + 8), fill=(218, 218, 0))
        
        bird.save(f'assets/bird{i}.png')
    
    # Create a simple pipe
    pipe = Image.new('RGBA', (52, 320), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pipe)
    
    # Main pipe body
    draw.rectangle((0, 0, 52, 320), fill=(0, 128, 0))
    
    # Pipe cap
    draw.rectangle((-5, 0, 57, 20), fill=(0, 150, 0))
    
    # Highlight
    draw.rectangle((5, 25, 15, 320), fill=(0, 150, 0))
    
    pipe.save('assets/pipe.png')
    
    # Create a simple background
    bg = Image.new('RGB', (288, 512), (135, 206, 250))
    draw = ImageDraw.Draw(bg)
    
    # Draw clouds
    for i in range(10):
        x = random.randint(0, 288)
        y = random.randint(0, 350)
        size = random.randint(20, 60)
        draw.ellipse((x, y, x+size, y+size//2), fill=(255, 255, 255))
        draw.ellipse((x-10, y+5, x+size-10, y+size//2+5), fill=(255, 255, 255))
        draw.ellipse((x+10, y+5, x+size+10, y+size//2+5), fill=(255, 255, 255))
    
    bg.save('assets/background.png')
    
    # Create a simple base/ground
    base = Image.new('RGB', (336, 112), (222, 216, 149))
    draw = ImageDraw.Draw(base)
    
    # Top border
    draw.rectangle((0, 0, 336, 10), fill=(171, 154, 96))
    
    # Texture lines
    for i in range(20):
        y = random.randint(15, 100)
        width = random.randint(20, 100)
        x = random.randint(0, 336 - width)
        draw.rectangle((x, y, x + width, y + 2), fill=(200, 188, 120))
    
    base.save('assets/base.png')
    
    print("Assets generated successfully in the 'assets' directory!")
    print("You can replace these with proper game assets if desired.")

if __name__ == "__main__":
    generate_assets()