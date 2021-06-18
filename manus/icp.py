from matplotlib import pyplot as plt
import seaborn as sns

def center_data(data, exclude_indices=[]):
    reduced_data = np.delete(data, exclude_indices, axis=1)
    center = np.array([reduced_data.mean(axis=1)]).T
    return center, data - center


def compute_cross_covariance(P, Q, correspondences, kernel=lambda diff: 1.0):
    cov = np.zeros((3, 3))
    exclude_indices = []
    for i, j in correspondences:
        p_point = P[:, [i]]
        q_point = Q[:, [j]]
        weight = kernel(p_point - q_point)
        if weight < 0.01: exclude_indices.append(i)
        cov += weight * q_point.dot(p_point.T)
    return cov, exclude_indices


def get_correspondence_indices(P, Q):
    p_size = P.shape[1]
    q_size = Q.shape[1]
    correspondences = []

    for i in range(p_size):
        correspondences.append((i, i))

    return correspondences


def plot_values(values, label):
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100
    sns.set_theme()
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(values[:2], label=label)
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0)
    ax.set_xlabel("")
    plt.show()


def plot_bar_values(values):
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100
    sns.set_theme()
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.bar(["Before ICP", "After ICP"], values[:2])
    ax.grid(True)
    ax.set_ylim(0)
    ax.set_ylabel("Distance")
    ax.set_title("Distance between pointclouds")
    plt.show()