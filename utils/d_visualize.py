import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def show_voxel_grid(voxel_tensor, threshold=0.5, title="3D Voxel Grid"):
    """Display a voxel grid in a 3D matplotlib window."""
    voxel_data = voxel_tensor.squeeze().cpu().numpy()
    filled = voxel_data > threshold
    x, y, z = np.where(filled)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, zdir='z', c='blue', marker='o', alpha=0.5, s=10)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()  # This opens a real-time interactive 3D window
