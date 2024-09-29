import time
from get_input_args import get_input_args
from get_pet_labels import get_pet_labels
from classify_images import classify_image
from check_images import check_images
from adjust_results4_isadog import adjust_results4_isadog
from calculates_results_stats import calculates_results_stats
from print_results import print_results
from torchvision import models

def main():
    # Start timing
    start_time = time.time()

    # Get input arguments
    in_arg = get_input_args()

    # Check images
    valid_images = check_images(in_arg.dir)  # Validate images in the directory
    if not valid_images:
        print("No valid images found. Exiting.")
        return  # Exit if no valid images are found

    # Load the pre-trained model
    model = models.vgg16(pretrained=True)  # Load VGG model

    # Get pet labels
    pet_labels = get_pet_labels(in_arg.dir)

    # Classify images
    results = {}
    for filename in pet_labels.keys():
        if filename in valid_images:  # Only classify valid images
            image_path = os.path.join(in_arg.dir, filename)  # Use os.path.join for compatibility
            predicted_label_idx = classify_image(image_path, model)
            results[filename] = [predicted_label_idx]  # Store predicted label index

    # Adjust results for dog classification
    results = adjust_results4_isadog(results, in_arg.dogfile)

    # Calculate overall accuracy
    accuracy = calculates_results_stats(results)

    # Print results
    print_results(results, accuracy)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
