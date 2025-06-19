import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_two_voxels(merged_voxels,  threshold=0.5, save_path=None):
    """
    Plots side-by-side 3D scatter plots of merger and refiner voxel outputs.

    Args:
        merged_voxels (torch.Tensor or np.ndarray): [D, H, W] or [1, D, H, W]
        refined_voxels (torch.Tensor or np.ndarray): [D, H, W] or [1, D, H, W]
        threshold (float): Threshold for binarization.
        save_path (str): If given, saves the plot.
    """
    if hasattr(merged_voxels, "detach"):
        merged_voxels = merged_voxels.detach().cpu().numpy()
    # if hasattr(refined_voxels, "detach"):
    #     refined_voxels = refined_voxels.detach().cpu().numpy()

    # Remove batch dimension if present
    if merged_voxels.ndim == 4:
        merged_voxels = merged_voxels[0]
    # if refined_voxels.ndim == 4:
    #     refined_voxels = refined_voxels[0]

    # Binarize
    merged_bin = (merged_voxels > threshold)
    # refined_bin = (refined_voxels > threshold)

    # Get nonzero coordinates
    merged_coords = np.array(np.nonzero(merged_bin))
    # refined_coords = np.array(np.nonzero(refined_bin))

    fig = plt.figure(figsize=(8, 8))

    # Merger output
    ax1 = fig.add_subplot(121, projection='3d')
    if merged_coords.shape[1] > 0:
        ax1.scatter(merged_coords[0], merged_coords[1], merged_coords[2], c='blue', alpha=0.5, s=10)
    ax1.set_title(f"Merger Output\n({merged_bin.sum()} voxels)", fontsize=10)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # # Refiner output
    # ax2 = fig.add_subplot(122, projection='3d')
    # if refined_coords.shape[1] > 0:
    #     ax2.scatter(refined_coords[0], refined_coords[1], refined_coords[2], c='red', alpha=0.5, s=10)
    # ax2.set_title(f"Refiner Output\n({refined_bin.sum()} voxels)", fontsize=10)
    # ax2.set_xlabel('X')
    # ax2.set_ylabel('Y')
    # ax2.set_zlabel('Z')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()