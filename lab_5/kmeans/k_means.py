import numpy as np

def initialize_centroids_forgy(data, k):
    # TODO implement random initialization
    return np.random.choice(data, k, replace=False)

def initialize_centroids_kmeans_pp(data, k):
    # TODO implement kmeans++ initizalization

    return None

def assign_to_cluster(data, centroids):
    # Find the closest cluster for each data point
    distances = np.sqrt(np.sum((data[:, np.newaxis] - centroids)**2, axis=2))
    return np.argmin(distances, axis=1)

def update_centroids(data, assignments, centroids):
    # Find new centroids based on the assignments
    new_centroids = np.array([data[assignments == k].mean(axis=0) for k in range(centroids.shape[0])])
    return new_centroids

def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments])**2))

def k_means(data, num_centroids, kmeansplusplus=False):
    # Centroids initialization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else:
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments = assign_to_cluster(data, centroids)
    for i in range(100):  # Max number of iterations = 100
        #print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments, centroids)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments):  # Stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)
