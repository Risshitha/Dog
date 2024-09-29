def print_results(results, accuracy):
    """
    Prints the results of the classification in a readable format.
    """
    for filename, labels in results.items():
        print(f"{filename}: Predicted: {labels[0]}, Dog: {'Yes' if labels[2] == 1 else 'No'}")
    print(f"\nOverall accuracy: {accuracy:.2f}")
