import numpy as np
from random import choice, sample
from math import sqrt




def initialize_centroids_forgy(data: np.array, k = 3):
    # TODO implement random initialization
    return sample(population=data.tolist(), k=k )

def initialize_centroids_kmeans_pp(data, k = 3):
    # TODO implement kmeans++ initizalization
    centroids = []
    centroids.append(choice(data.tolist()))
    for _ in range(k - 1):
        furthest_dist = -np.inf
        furthest_point = None
        for obs in data:
            distance = sum(sum(pow(centroids - obs, 2)))
            if distance > furthest_dist:
                furthest_dist = distance
                furthest_point = obs
        centroids.append(furthest_point)    
    return centroids

def assign_to_cluster(data, centroids):
    # TODO find the closest cluster for each data point
    def corelate_to_centroid(point,centroids):
        min_dist = np.inf
        min_centr_ind = None
        for centr_ind, centroid in enumerate(centroids):
            dist = sum(pow(point - centroid, 2))
            if dist < min_dist:
                min_dist = dist
                min_centr_ind = centr_ind
        return min_centr_ind
    
    return np.array(list(map(lambda x:corelate_to_centroid(x, centroids), data)))

def update_centroids(data, assignments):
    # TODO find new centroids based on the assignments
    clusters = [[] for _ in range(max(assignments) + 1)]
    for i, assignment in enumerate(assignments):
        clusters[assignment].append(data[i])

    return np.array([np.mean(cluster, axis=0) for cluster in clusters if cluster])


def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - np.take(centroids, assignments, axis=0))**2))


def k_means(data, num_centroids, kmeansplusplus= False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)

    
    assignments  = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return assignments, centroids, mean_intra_distance(data, assignments, centroids)         

