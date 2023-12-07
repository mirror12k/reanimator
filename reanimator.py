#!/usr/bin/env python3
import argparse
import sys
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
from collections import Counter



def scale_image(image, new_width=100):
    width, height = image.size
    aspect_ratio = height / width
    new_height = int(aspect_ratio * new_width)
    resized_image = image.resize((new_width, new_height))
    return resized_image

def scale_image_doublewidth(image, new_width=100):
    width, height = image.size
    aspect_ratio = height / width
    new_height = int(aspect_ratio * new_width)
    resized_image = image.resize((new_width*2, new_height))
    return resized_image

def is_background_color(rgb_pixel, threshold=250):
    # Determine if the pixel color is close to white (or background color)
    return all(c > threshold for c in rgb_pixel)

def get_background_color(image):
    # Analyze the pixels at the borders to find the most common color
    width, height = image.size
    pixels = []
    for x in range(width):
        pixels.append(image.getpixel((x, 0)))  # Top
        pixels.append(image.getpixel((x, height - 1)))  # Bottom
    for y in range(height):
        pixels.append(image.getpixel((0, y)))  # Left
        pixels.append(image.getpixel((width - 1, y)))  # Right

    most_common = Counter(pixels).most_common(1)
    return most_common[0][0] if most_common else (255, 255, 255)

def apply_vertical_kernel(image, kernel_size=3):
    # Ensure kernel size is odd for symmetry
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create a new image to store the results
    processed_image = Image.new("RGB", image.size)
    pixels = image.load()
    new_pixels = processed_image.load()

    width, height = image.size

    # Apply the vertical kernel
    for x in range(width):
        for y in range(height):
            vertical_sum = [0, 0, 0]
            for ky in range(-(kernel_size // 2), kernel_size // 2 + 1):
                pixel_y = max(0, min(y + ky, height - 1))
                for c in range(3):  # Iterate over RGB channels
                    vertical_sum[c] += pixels[x, pixel_y][c]
            
            # Average the sum and set the pixel
            new_pixels[x, y] = tuple(int(sum(c) / kernel_size) for c in zip(vertical_sum))

    return processed_image

def apply_horizontal_kernel(image, kernel_size=3):
    # Ensure kernel size is odd for symmetry
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create a new image to store the results
    processed_image = Image.new("RGB", image.size)
    pixels = image.load()
    new_pixels = processed_image.load()

    width, height = image.size

    # Apply the horizontal kernel
    for x in range(width):
        for y in range(height):
            horizontal_sum = [0, 0, 0]
            for kx in range(-(kernel_size // 2), kernel_size // 2 + 1):
                pixel_x = max(0, min(x + kx, width - 1))
                for c in range(3):  # Iterate over RGB channels
                    horizontal_sum[c] += pixels[pixel_x, y][c]
            
            # Average the sum and set the pixel
            new_pixels[x, y] = tuple(int(sum(c) / kernel_size) for c in zip(horizontal_sum))

    return processed_image

def map_pixel_to_ascii_color(rgb_pixel, background_color, ascii_palette, is_grayscale=False):
    if rgb_pixel == background_color:
        return ' ', (0, 0, 0)  # Return space for background

    if is_grayscale:
        # For grayscale, use the standard mapping
        gray = int((rgb_pixel[0] + rgb_pixel[1] + rgb_pixel[2]) / 3)
        ascii_char = ascii_palette[gray * len(ascii_palette) // 256]
        return ascii_char, gray

    gray = int((rgb_pixel[0] + rgb_pixel[1] + rgb_pixel[2]) / 3)
    ascii_char = ascii_palette[gray * len(ascii_palette) // 256]
    return ascii_char, rgb_pixel

def apply_kernel(image, kernel):
    if len(kernel) != 9:
        raise ValueError("Kernel must have 9 elements")

    # Create a new image to store the results
    processed_image = Image.new("L", image.size)
    pixels = image.load()
    new_pixels = processed_image.load()
    kernel_sum = sum(kernel)

    width, height = image.size

    # Apply the kernel
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            weighted_sum = 0
            k = 0
            for ky in range(-1, 2):
                for kx in range(-1, 2):
                    pixel_value = pixels[x + kx, y + ky]
                    weighted_sum += pixel_value * kernel[k]
                    k += 1

            # Set the new pixel value, ensuring it's within valid range [0, 255]
            new_value = min(max(int(weighted_sum / kernel_sum), 0), 255)
            new_pixels[x, y] = new_value

    return processed_image

def select_max_kernel_char(x, y, kernel_images, characters, dominance_threshold):
    # Get the pixel value from each kernel-processed image
    values = [img.getpixel((x, y)) for img in kernel_images]

    # Find the maximum value and its index
    max_value = max(values)
    max_index = values.index(max_value)

    # Check if the maximum value is dominant enough
    if max_value < dominance_threshold:
        return ' '  # Return space if no kernel is dominant
    else:
        return characters[max_index]

def apply_threshold_to_image(image, threshold):
    thresholded_image = Image.new("L", image.size)
    pixels = image.load()
    new_pixels = thresholded_image.load()

    for x in range(image.size[0]):
        for y in range(image.size[1]):
            new_pixels[x, y] = 0 if pixels[x, y] < threshold else pixels[x, y]

    return thresholded_image

def create_ascii_art_from_kernels(original_image, kernel_images, characters, dominance_threshold):
    width, height = original_image.size
    ascii_art = []

    for y in range(height):
        row = ""
        for x in range(width):
            char = select_max_kernel_char(x, y, kernel_images, characters, dominance_threshold)
            row += char
        ascii_art.append(row)
    
    return ascii_art



def generate_kernelized_ascii_arg(image, threshold=75, dominance_threshold=60):
    # Define your kernels and corresponding characters
    kernels = [
        [0,0,0,1,1,1,0,0,0],      # '-'
        [0,1,0,0,1,0,0,1,0],      # '|'
        [1,0,0,0,1,0,0,0,1],      # '\\'
        [0,0,1,0,1,0,1,0,0],      # '/'
        [0,1,0,1,1,1,0,1,0],      # '+'
        # [0,0,0,0,1,0,1,0,1],      # '^'
        # [1,0,1,0,1,0,0,0,0],      # 'v'
        # [1,0,1,0,1,0,1,0,1],      # 'x'
        # [1,1,1,1,1,1,1,1,1],      # '#'
    ]
    characters = "-|\\/+"

    # Apply each kernel to the image
    kernel_images = [apply_kernel(image, k) for k in kernels]
    kernel_images_thresholded = [apply_threshold_to_image(img, threshold) for img in kernel_images]

    # Create ASCII art from the kernel images
    ascii_art = create_ascii_art_from_kernels(image, kernel_images_thresholded, characters, dominance_threshold)
    return ascii_art

def apply_color_to_edges(original_image, edge_image):
    # Convert the original image to 'RGBA' to get the color information
    color_image = original_image.convert('RGBA')

    # Convert the edge-detected image to 'RGBA' and change its color to black (or any color of choice)
    # so that only the alpha channel has the edge information
    edge_image_rgba = edge_image.convert('RGBA')
    black_image = Image.new('RGBA', edge_image_rgba.size, (0, 0, 0, 255))  # Black image
    edges_colored = Image.blend(black_image, edge_image_rgba, alpha=1)

    # Combine the colored edges with the original image
    combined_image = Image.alpha_composite(color_image, edges_colored)

    return combined_image

import colorsys

def apply_hue_shift(image, degree):
    # Convert degree to a value between 0 and 1
    shift = degree / 360.0

    # Load the image data
    image = image.convert('RGB')
    pixels = image.load()
    width, height = image.size

    # Process each pixel
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]

            # Convert RGB to HSV
            h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

            # Adjust Hue value and ensure it remains within [0, 1]
            h = (h + shift) % 1.0

            # Convert HSV back to RGB
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            pixels[x, y] = int(r * 255), int(g * 255), int(b * 255)

    return image

def apply_rainbow_hue_shift(image, degree_multiplier):
    image = image.convert('RGB')
    pixels = image.load()
    width, height = image.size

    # Process each pixel
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]

            # Convert RGB to HSV
            h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

            # Calculate the hue shift based on x and y coordinates
            shift = ((x + y) * degree_multiplier) % 360
            shift /= 360.0  # Normalize to [0, 1]

            # Adjust Hue value and ensure it remains within [0, 1]
            h = (h + shift) % 1.0

            # Convert HSV back to RGB
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            pixels[x, y] = int(r * 255), int(g * 255), int(b * 255)

    return image

def apply_median_blur(image, size=3):
    if size % 2 == 0:
        raise ValueError("Size must be an odd number")

    return image.filter(ImageFilter.MedianFilter(size))


def generate_rgb_ascii_art(image, background_color, ascii_palette, is_grayscale):
    # Convert to ASCII
    ascii_art = []
    for y in range(image.height):
        row = []
        for x in range(image.width):
            pixel = image.getpixel((x, y))
            ascii_char, color = map_pixel_to_ascii_color(pixel, background_color, ascii_palette, is_grayscale)
            row.append((ascii_char, color))
        ascii_art.append(row)
    
    return ascii_art

def convert_to_ascii_art(image,
            text_width=50,
            ascii_palette='░▒▓█',
            is_grayscale=False,
            kernelize_image=False,
            keep_background=False,
            use_edge_detection=False,
            hue_shift=0,
            rainbow_hue_shift=0,
            median_blur=0,
            gradient_strength=1.0):
    # Load the image using Pillow
    image = image.convert("RGB")

    # Convert to grayscale if option is set
    if is_grayscale:
        image = image.convert("L").convert("RGB")

    # Find the background color
    background_color = -1 if keep_background else get_background_color(image)

    # Resize image
    resized_image = scale_image_doublewidth(image, int(text_width/2))
    image = resized_image

    # Optionally perform edge detection
    if use_edge_detection or kernelize_image:
        # Enhance the contrast of the image to control gradient strength
        enhancer = ImageEnhance.Contrast(image)
        enhanced_image = enhancer.enhance(gradient_strength)

        # Apply edge detection filter
        image = enhanced_image.filter(ImageFilter.FIND_EDGES)

        # Crop the outer line of pixels
        width, height = image.size
        image = image.crop((1, 1, width - 1, height - 1))
        # Crop the outer line of pixels
        width, height = resized_image.size
        original_image = resized_image.crop((1, 1, width - 1, height - 1))

        image = apply_color_to_edges(original_image, image)

    if hue_shift != 0:
        image = apply_hue_shift(image, hue_shift)

    if rainbow_hue_shift != 0:
        image = apply_rainbow_hue_shift(image, rainbow_hue_shift)

    if median_blur != 0:
        image = apply_median_blur(image, median_blur)

    if kernelize_image:
        image = image.convert("L")
        return generate_kernelized_ascii_arg(image)
    else:
        # # Convert to RGB before saving to remove the alpha channel (if present)
        image = image.convert('RGB')
        return generate_rgb_ascii_art(image, background_color, ascii_palette, is_grayscale)


def print_colored_ascii_art(ascii_art, is_grayscale=False):
    if type(ascii_art[0]) is str:
        for line in ascii_art:
            print(line)
    else:
        for row in ascii_art:
            for char, color in row:
                if is_grayscale or char == ' ':
                    print(char, end="")
                else:
                    print(f"\033[38;2;{color[0]};{color[1]};{color[2]}m{char}\033[39m", end="")
            print()

def print_colored_ascii_art_html(ascii_art, is_grayscale=False):
    html_output = '<pre style="font: monospace; background-color: black; color: white;">'

    if type(ascii_art[0]) is str:
        for line in ascii_art:
            html_output += line + '\n'
    else:
        for row in ascii_art:
            for char, color in row:
                if is_grayscale or char == ' ':
                    html_output += char
                else:
                    html_output += f'<span style="color: rgb({color[0]},{color[1]},{color[2]});">{char}</span>'
            html_output += '\n'

    html_output += '</pre>'
    print(html_output)

def text_to_image(text, font_path='arial.ttf', font_size=80, text_color=(0, 0, 0), bg_color=(255, 255, 255)):
    # Load the font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
        print("Default font will be used as specified font was not found.")

    # Create a dummy image to calculate text size
    dummy_image = Image.new('RGB', (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_image)

    # Calculate size of the text
    text_width = int(max( dummy_draw.textlength(line, font=font, font_size=80) for line in text.split('\n') ))
    text_height = int(font_size * 0.1 + font_size * len(text.split('\n')))

    # Create an image with the correct size and background color
    image = Image.new('RGB', (text_width, text_height), bg_color)
    draw = ImageDraw.Draw(image)

    # Draw the text onto the image
    draw.text((0, 0), text, fill=text_color, font=font, font_size=80)

    return image

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def print_logo():
    try:
        with open('logo.txt', 'r') as file:
            print(file.read())
    except FileNotFoundError:
        print(f"Logo not found?")

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Convert text or images to ASCII art. Specify an image path or a a --text argument to generate ascii art.", add_help=False)
    
    # Add arguments
    parser.add_argument("--greyscale", action="store_true", help="Render the image in greyscale")
    parser.add_argument("--width", type=int, default=64, help="Width of the ASCII art in characters, default is 64")
    parser.add_argument("--palette", type=str, default=' .,:;-=s$#@', help="ASCII palette to use for rendering")
    parser.add_argument("--palette-block", action="store_true", help="Use the block ASCII palette '░▒▓█'")
    parser.add_argument("--palette-ascii", action="store_true", help="Use the extended ASCII palette '.-;=s$#@'")
    parser.add_argument("--keep-background", action="store_true", help="Keep the background in the ASCII art")
    parser.add_argument("--kernelize-image", action="store_true", help="Apply edge detection and attempt to draw the edges with lines")
    parser.add_argument("--gradient", action="store_true", help="Apply edge detection to the image")
    parser.add_argument("--gradient-strength", type=float, default=1.0, help="Control the strength of the gradient in edge detection")
    parser.add_argument("--hue-shift", type=int, default=0, help="Hue-shifts the image before transforming it")
    parser.add_argument("--rainbow-hue-shift", type=float, default=0, help="Hue-shifts the image in rainbow pattern at the speed controlled by this argument")
    parser.add_argument("--median-blur", type=int, default=0, help="Median blurs the image with the given kernel size")
    parser.add_argument("--text", type=str, help="Generates an image of the text and then converts it to ascii art")
    parser.add_argument("--text-font", type=str, default='font/Roboto-Regular.ttf', help="The font used in text to image generation, defaults to roboto")
    parser.add_argument("--text-size", type=int, default=80, help="The font size used in text to image generation, defaults to 80")
    parser.add_argument("--text-color", type=str, default='#ff40af', help="Color of the text to image, must be a hex color code like '#ff03a1'")
    parser.add_argument("--html-output", action="store_true", help="Outputs the image as an html block rather than terminal text")
    parser.add_argument("image", nargs="*", help="List of images to convert")

    parser.add_argument('--help', action='store_true', help='Show help message and exit.')

    # Parse arguments
    args = parser.parse_args()

    # Determine the ASCII palette
    if args.palette_block:
        ascii_palette = ' ░▒▓█'
    elif args.palette_ascii:
        ascii_palette = '.-;=s$#@'
    else:
        ascii_palette = args.palette

    if args.help or len(sys.argv) == 1:
        print_logo()
        parser.print_help()
        return

    image_objects = [ Image.open(image_path) for image_path in args.image ]

    if args.text:
        image = text_to_image(args.text, font_path=args.text_font, font_size=args.text_size, text_color=hex_to_rgb(args.text_color), bg_color=(0, 0, 0))
        image_objects.append(image)

    # Process each image
    for image in image_objects:
        ascii_art = convert_to_ascii_art(image,
                text_width=args.width,
                ascii_palette=ascii_palette,
                is_grayscale=args.greyscale,
                kernelize_image=args.kernelize_image,
                keep_background=args.keep_background,
                hue_shift=args.hue_shift,
                rainbow_hue_shift=args.rainbow_hue_shift,
                median_blur=args.median_blur,
                use_edge_detection=args.gradient,
                gradient_strength=args.gradient_strength)
        if args.html_output:
            print_colored_ascii_art_html(ascii_art, is_grayscale=args.greyscale)
        else:
            print_colored_ascii_art(ascii_art, is_grayscale=args.greyscale)

if __name__ == "__main__":
    main()


