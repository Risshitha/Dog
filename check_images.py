import os

def check_images(images_dir):
    """
    Check if all images in the directory exist and are valid images.
    Returns a list of valid image file names.
    """
    valid_images = []
    for file in os.listdir(images_dir):
        if file.endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(os.path.join(images_dir, file)):
            valid_images.append(file)
        else:
            print(f"Invalid file format: {file}")
    return valid_images
