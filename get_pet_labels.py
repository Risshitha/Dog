import os

def get_pet_labels(images_dir):
    """
    Creates a dictionary of pet labels based on image filenames.
    The dictionary will map the filename to a list containing the pet label.
    """
    pet_labels = {}
    for filename in os.listdir(images_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Extract pet label from filename and format it
            pet_label = filename.split('_')[0].lower().strip()  # Example: "poodle_07956.jpg" => "poodle"
            pet_labels[filename] = [pet_label]  # Store the label as a list
    return pet_labels
