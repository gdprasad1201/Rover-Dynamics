import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Plot each set of obstacles
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns for 6 plots
    fig.suptitle("Obstacle Sets on a 2.5x2.5 Field", fontsize=16)

    field_size = 2.5

    obstacles1 = [[1.07, 1.93, 0.09], [0.35, 1.25, 0.24], [0.29, 0.5, 0.24], [0.25, 1.71, 0.26], [1.63, 1.64, 0.06]]
    obstacles2 = [[1.7,1.78,0.24],[0.97,1.78,0.17],[0.39,2.27,0.15],[0.07,2.07,0.11],[1.49,1.25,0.23]]
    obstacles3 = [[1.86,1.00,0.27],[2.19,2.3,0.09],[0.43,0.05,0.23],[1.38,2.15,0.18]]
    obstacles4 = [[0.14,1.17,0.14],[2.20,0.95,0.16]]
    obstacles5 = [[0.47,2.15,0.2],[0.72,1.18,0.08],[1.44,0.5,0.19]]
    obstacles6 = [[0.65,0.15,0.1],[0.23,2.25,0.13],[1.08,0.84,0.17],[0.84,2.18,0.12],[1.94,1.53,0.27]]

    obstacle_list = [obstacles1, obstacles2, obstacles3, obstacles4, obstacles5, obstacles6]

    for i, obstacles in enumerate(obstacle_list):
        ax = axs[i // 3, i % 3]  # Determine subplot location
        ax.set_xlim(-0.5, field_size)
        ax.set_ylim(-0.5, field_size)
        ax.set_title(f"Set {i + 1}", fontsize=12)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal', adjustable='box')

        # Plot each obstacle as a circle
        for obstacle in obstacles:
            circle = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='blue', alpha=0.5)
            ax.add_patch(circle)

        # Add grid for better visualization
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
    plt.show()
