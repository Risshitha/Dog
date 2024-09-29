import argparse

def get_input_args():
    """
    Retrieves and parses the command line arguments provided by the user
    when they run the program.
    Returns these arguments as an object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='pet_images/', help='Directory of pet images')
    parser.add_argument('--arch', type=str, default='vgg', help='Model architecture (default: vgg)')
    parser.add_argument('--dogfile', type=str, default='dognames.txt', help='File with dog names')
    return parser.parse_args()
