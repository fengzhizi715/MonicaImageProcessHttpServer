#!/usr/bin/env python3
"""
Create a simple test image for the image processing server tests
"""

from PIL import Image, ImageDraw
import numpy as np

def create_test_image():
    # Create a simple test image (256x256 RGB image with shapes)
    img = Image.new('RGB', (256, 256), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    
    # Draw a yellow rectangle
    d.rectangle([50, 50, 200, 150], fill=(255, 255, 0))
    
    # Draw a red ellipse
    d.ellipse([100, 100, 200, 200], fill=(255, 0, 0))
    
    # Draw a blue line
    d.line([0, 0, 256, 256], fill=(0, 0, 255), width=5)
    
    # Save the image
    img.save('test_image.jpg', 'JPEG')
    print("Created test_image.jpg")

if __name__ == "__main__":
    create_test_image()