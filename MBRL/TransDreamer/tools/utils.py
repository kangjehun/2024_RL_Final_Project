# Functions for print
def print_colored(message, color):
    colors = {
        # Normal colors
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        # Dark colors
        "dark_red": "\033[31m",
        "dark_green": "\033[32m",
        "dark_yellow": "\033[33m",
        "dark_blue": "\033[34m",
        "dark_purple": "\033[35m",
        "dark_cyan": "\033[36m",
        "dark_white": "\033[37m",
        # Bright colors
        "bright_red": "\033[91;1m",
        "bright_green": "\033[92;1m",
        "bright_yellow": "\033[93;1m",
        "bright_blue": "\033[94;1m",
        "bright_purple": "\033[95;1m",
        "bright_cyan": "\033[96;1m",
        "bright_white": "\033[97;1m",
        "end": "\033[0m",
    }
    print(f"{colors.get(color, colors['end'])}{message}{colors['end']}")   
    
def print_centered_message(message, border_char, total_width, color=None):
    """ Print a message centered within a border of a specific character """
    border_length = total_width
    message_length = len(message)
    padding = (border_length - message_length) // 2
    if padding > 0:
        line = border_char * padding + message + border_char * padding
        # If the total width isn't even, add an extra border character to the right
        if len(line) < total_width:
            line += border_char
    else:
        line = message
    if color is not None:
        print_colored(line, color)
    else:
        print(line)    
    
