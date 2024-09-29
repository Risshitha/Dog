def calculates_results_stats(results):
    """
    Calculates the results statistics such as the number of correct classifications.
    """
    correct = 0
    total = len(results)
    
    for _, value in results.items():
        if value[2] == 1:  # Check if the classified label is a dog
            correct += 1  # Increment correct if it is classified correctly
    
    accuracy = correct / total if total > 0 else 0
    return accuracy
