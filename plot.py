import pandas as pd
import matplotlib.pyplot as plt
# Used csv file:
# csv_logs/success_rate_curve_20260405_120943.csv    ----2 million, ent_cof = 0 (important)
# csv_logs/success_rate_curve_20260405_215143.csv    ----2 million, ent_cof = 0.01

def plot_single_curve(
    csv_path,
    x_column,
    y_column,
    title,
    x_label,
    y_label,
    output_path=None
):
    """
    Read a CSV file and plot a single curve.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    x_column : str
        Column name for the x-axis.
    y_column : str
        Column name for the y-axis.
    title : str
        Figure title.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.
    output_path : str or None
        If provided, save the figure to this path.
    """

    # Read CSV data
    df = pd.read_csv(csv_path)

    # Check whether required columns exist
    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' not found in CSV.")
    if y_column not in df.columns:
        raise ValueError(f"Column '{y_column}' not found in CSV.")

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Plot the curve
    plt.plot(df[x_column], df[y_column], marker='o', markersize=4, linewidth=2)

    # Add title and axis labels
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=13)
    plt.ylabel(y_label, fontsize=13)

    # Add grid for readability
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
    # Replace the CSV path and column names with your own
    plot_single_curve(
        csv_path="csv_logs/success_rate_curve_20260405_120943.csv",
        x_column="Timesteps",
        y_column="Success Rate",
        title="Training Success Rate of PPO (without Entropy) over 2 Million Timesteps",
        x_label="Timesteps",
        y_label="Success Rate (%)",
        output_path="pictures/ppo_success_rate.png"
    )