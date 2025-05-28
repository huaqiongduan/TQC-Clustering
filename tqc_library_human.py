
import numpy as np

def cluster_fitness(indexes, samples):  # Eq. 8
    total = 0
    for ids in indexes:
        pts = samples[ids]
        if len(pts) > 0:
            m = np.mean(pts, axis=0)
            total += np.sum(np.linalg.norm(pts - m, axis=1))
    return total

def initialize_centroids(data_set, num_clust):  # Eq. 9
    return data_set[np.random.choice(data_set.shape[0], num_clust, replace=False)]

def assign_labels(data_arr, centroids):
    lab = []
    for pt in data_arr:
        d = [np.linalg.norm(pt - c) for c in centroids]
        lab.append(np.argmin(d))
    return np.array(lab)

def global_outlier_indices(cluster_lists, xdata, threshold=2):  # Eq. 10-12
    outliers = []
    remain = []
    for i, cluster_idx in enumerate(cluster_lists):
        if len(cluster_idx) > threshold:
            ref = np.mean(xdata[cluster_idx], axis=0)
            dists = np.linalg.norm(xdata[cluster_idx] - ref, axis=1)
            mean_dist = np.mean(dists)
            for idx, dist in zip(cluster_idx, dists):
                if dist > mean_dist:
                    outliers.append(idx)
                else:
                    remain.append(idx)
    return outliers, remain

def select_seedlings(sorted_groups, fitness_array):  # Eq. 13
    n_sel = max(1, len(sorted_groups) // 2)
    idx_fit = np.argsort(fitness_array)
    return [sorted_groups[i] for i in idx_fit[:n_sel]]

def virtual_merge(a, b):  # Eq. 14
    return 0.5 * (a + b)

def assign_new_member(data_set, candidate, all_centroids):  # Eq. 15
    dists = [np.linalg.norm(candidate - c) for c in all_centroids]
    return np.argmin(dists)

def growth_ratio(fit_now, fit_prev):  # Eq. 16
    return fit_now / (fit_prev + 1e-8)

def seedling_proliferate(seed_arr, nonsd_arr, base_group, w=0.3):  # Eq. 17-19
    improved = []
    for s, n in zip(seed_arr, nonsd_arr):
        diff = w * (s - n)
        new = n + diff
        fit_orig = cluster_fitness([base_group], np.vstack((s, n)))
        fit_new = cluster_fitness([base_group], np.vstack((s, new)))
        if fit_new < fit_orig:
            improved.append(new)
        else:
            improved.append(n)
    return np.array(improved)

def layer_mutation(x, eps=0.13):  # Eq. 20
    rand = np.random.uniform(-1, 1, x.shape)
    return x + eps * rand

def update_clusters(clusters, fitness_list, max_size):  # Eq. 21
    if len(clusters) > max_size:
        idx = np.argsort(fitness_list)[:max_size]
        clusters = [clusters[i] for i in idx]
    return clusters

def tqc_algorithm(all_clusters, X, num_epochs, k_group):  # Algorithms 2 & 3, Eqs 8–22
    Qtbl = np.zeros((len(all_clusters), k_group))
    lr, discount = 0.09, 0.93
    fit_hist = []
    prev_fitness = cluster_fitness(all_clusters, X)
    for ep in range(num_epochs):
        curr_fitness = cluster_fitness(all_clusters, X)
        fit_hist.append(curr_fitness)
        out_idx, kept_idx = global_outlier_indices(all_clusters, X)
        fit_array = np.array([cluster_fitness([ids], X) for ids in all_clusters])
        seedlings = select_seedlings(all_clusters, fit_array)
        gr = growth_ratio(curr_fitness, prev_fitness)
        reward = -curr_fitness
        for j in range(len(all_clusters)):
            action = np.random.randint(0, k_group)
            Qtbl[j, action] += lr * (reward + discount * np.max(Qtbl[j]) - Qtbl[j, action])
        prev_fitness = curr_fitness
    return all_clusters, Qtbl, fit_hist
