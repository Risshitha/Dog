def adjust_results4_isadog(results, dogfile):
    """
    Adjusts the results dictionary to indicate if each classified label is a dog or not.
    """
    with open(dogfile, 'r') as f:
        dog_names = {line.strip().lower() for line in f.readlines()}  # Set of dog names

    for filename, prediction in results.items():
        is_dog = 1 if prediction in dog_names else 0  # 1 for dog, 0 for not a dog
        results[filename].append(is_dog)  # Append the dog status to results
    return results
