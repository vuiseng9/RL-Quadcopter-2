import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3d_episode_trajectory(df):
    df = df.reset_index(drop=True)
    # environment boundary
    axis_limits=[(-50, 50), (-50, 50), (0, 150)]

    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=10, pad=8)
    ax.set_xlabel("x", fontsize=16, labelpad=16)
    ax.set_ylabel("y", fontsize=16, labelpad=16)
    ax.set_zlabel("z", fontsize=16, labelpad=16)
    
    # Plot simulated points of trajectory
    ax.scatter(df.x, df.y, df.z)
    ax.set_title('Episode %d, Score: %0.2f' % (df.episode[0], df.reward.sum()))