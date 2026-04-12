import pandas as pd
import matplotlib.pyplot as plt


def plot_two_curves_two_csvs(
    csv_path_1,
    csv_path_2,
    x_column,
    y_column,
    label1,
    label2,
    title,
    x_label,
    y_label,
    output_path=None
):
    """
    Read two CSV files and plot one curve from each file for comparison.

    Parameters
    ----------
    csv_path_1 : str
        Path to the first CSV file.
    csv_path_2 : str
        Path to the second CSV file.
    x_column : str
        Column name for the x-axis in both CSV files.
    y_column : str
        Column name for the y-axis in both CSV files.
    label1 : str
        Legend label for the first curve.
    label2 : str
        Legend label for the second curve.
    title : str
        Figure title.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.
    output_path : str or None
        If provided, save the figure to this path.
    """

    # Read both CSV files
    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)

    # Check required columns
    for col in [x_column, y_column]:
        if col not in df1.columns:
            raise ValueError(f"Column '{col}' not found in first CSV.")
        if col not in df2.columns:
            raise ValueError(f"Column '{col}' not found in second CSV.")

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Plot the first curve
    plt.plot(df1[x_column], df1[y_column], marker='o', markersize=4, linewidth=2, label=label1)

    # Plot the second curve
    plt.plot(df2[x_column], df2[y_column], marker='s', markersize=4, linewidth=2, label=label2)

    # Add title and labels
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=13)
    plt.ylabel(y_label, fontsize=13)

    # Add legend and grid
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.5)

    # Adjust layout
    plt.tight_layout()

    # Save figure if needed
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Show figure
    plt.show()


if __name__ == "__main__":
    # Example usage:
    plot_two_curves_two_csvs(
        csv_path_1="ppo_750k.csv",
        csv_path_2="ppo_2m.csv",
        x_column="timesteps",
        y_column="success_rate",
        label1="PPO trained for 750k timesteps",
        label2="PPO trained for 2 million timesteps",
        title="Effect of Training Duration on Baseline PPO Success Rate",
        x_label="Timesteps",
        y_label="Success Rate (%)",
        output_path="training_duration_comparison.png"
    )