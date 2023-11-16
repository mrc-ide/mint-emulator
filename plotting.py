import numpy as np
import matplotlib.pyplot as plt
import torch
from dataset import MintDataset
import matplotlib.lines as mlines
from matplotlib.widgets import CheckButtons

def plot_mint_compare(predictions, actual, settings, num_samples=9):

    # Flatten the predictions and actual values into one-dimensional arrays
    flattened_predictions = np.concatenate(predictions, axis=0)
    flattened_actual = np.concatenate(actual, axis=0)

    # Select random row indices
    np.random.seed(0) # you can set any integer value
    row_indices = np.random.choice(flattened_predictions.shape[0], size=num_samples, replace=False)


    # Create a 3x3 grid of scatter plots
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Scatterplots of actual vs predicted values for selected time series')

    for i, row_index in enumerate(row_indices):
        # Get the selected row from the predictions and actual arrays
        selected_predictions = flattened_predictions[row_index]
        selected_actual = flattened_actual[row_index]

        ax = axs[i//3, i%3]
        ax.scatter(selected_actual, selected_predictions, c='black', alpha=0.5, marker='o', label='Predicted')
        ax.scatter(selected_actual, selected_actual, c='red', alpha=0.5, marker='o', label='Actual')
        ax.plot([np.min(selected_actual), np.max(selected_actual)], [np.min(selected_actual), np.max(selected_actual)], c='gray', linestyle='--')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'Scatterplot #{row_index}')
        ax.legend()

    plt.show()

def plot_mint_time_series(predictions, actual, settings, num_samples=9):
    
    # Flatten the predictions and actual values into one-dimensional arrays
    flattened_predictions = np.concatenate(predictions, axis=0)
    flattened_actual = np.concatenate(actual, axis=0)

    # Select random row indices
    np.random.seed(0) # you can set any integer value
    row_indices = np.random.choice(flattened_predictions.shape[0], size=num_samples, replace=False)


    # Create a 3x3 grid of time series plots
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Time series of actual vs predicted values for selected time series')

    for i, row_index in enumerate(row_indices):
        # Get the selected row from the predictions and actual arrays
        selected_predictions = flattened_predictions[row_index]
        selected_actual = flattened_actual[row_index]

        ax = axs[i//3, i%3]
        ax.plot(selected_predictions, c='black', label='Predictions', alpha=0.5, linestyle='-', marker='o', markersize=4)
        ax.plot(selected_actual, c='red', label='Actual', alpha=0.5, linestyle='-', marker='o', markersize=4)
        ax.set_xlabel('Year')
        ax.set_ylabel('Values')
        ax.set_title(f'Time series #{row_index}')

        # Adjust the x-axis tick labels
        tick_positions = np.arange(0, len(selected_predictions), 12)
        tick_labels = [2017 + i for i in tick_positions // 12]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        
        ax.legend()

    plt.show()
    
def plot_mint_avg_compare(predictions, actual, settings):

    # Flatten the predictions and actual values into one-dimensional arrays
    flattened_predictions = np.concatenate(predictions, axis=0)
    flattened_actual = np.concatenate(actual, axis=0)

    # Compute the average predicted and actual values
    avg_predicted = np.mean(flattened_predictions, axis=0)
    avg_actual = np.mean(flattened_actual, axis=0)

    # Create a scatter plot
    plt.figure(figsize=(7, 7))
    plt.scatter(avg_actual, avg_predicted, c='black', alpha=0.5, marker='o', label='Predicted')
    plt.scatter(avg_actual, avg_actual, c='red', alpha=0.5, marker='o', label='Actual')
    plt.plot([np.min(avg_actual), np.max(avg_actual)], [np.min(avg_actual), np.max(avg_actual)], c='gray', linestyle='--')
    plt.xlabel('Average Actual')
    plt.ylabel('Average Predicted')
    plt.title('Scatterplot of Average Actual vs Average Predicted')
    plt.legend()

    plt.show()

def plot_mint_avg_time_series(predictions, actual, settings):

    # Flatten the predictions and actual values into one-dimensional arrays
    flattened_predictions = np.concatenate(predictions, axis=0)
    flattened_actual = np.concatenate(actual, axis=0)

    # Compute the average predicted and actual values
    avg_predicted = np.mean(flattened_predictions, axis=0)
    avg_actual = np.mean(flattened_actual, axis=0)

    # Create a time series plot
    fig, ax = plt.subplots(figsize=(10, 7))

    lines = []

    # Plot the random trajectories and store them in a list
    random_lines = []

    num_samples = 100
    random_indices = np.random.choice(flattened_predictions.shape[0], size=num_samples, replace=False)
    for i in random_indices:
        random_lines.append(ax.plot(flattened_predictions[i], c='lightgray', alpha=0.5, linestyle='-', marker='^', markersize=2, visible=False)[0])
        random_lines.append(ax.plot(flattened_actual[i], c='darkgray', alpha=0.5, linestyle='--', marker='o', markersize=2, visible=False)[0])

    # Plot the average predicted and actual values
    lines.append(ax.plot(avg_predicted, c='black', label='Average Predictions', alpha=0.7, linestyle='-', marker='^', markersize=4)[0])
    lines.append(ax.plot(avg_actual, c='red', label='Average Actual', alpha=0.7, linestyle='-', marker='o', markersize=4)[0])

    # Adjust the x-axis tick labels and limits
    tick_positions = np.arange(0, len(avg_predicted), 12)
    tick_labels = [2017 + i for i in tick_positions // 12]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(tick_positions[0], tick_positions[-1])
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Values')
    ax.set_title('Time Series of Average Actual vs Average Predicted')

    # Create custom legend handles
    red_line = mlines.Line2D([], [], color='red', marker='o', linestyle='-', markersize=4, label='Average Actual')
    black_line = mlines.Line2D([], [], color='black', marker='^', linestyle='-', markersize=4, label='Average Predictions')
    ax.legend(handles=[red_line, black_line])

    # Create a CheckButtons widget to toggle the random trajectories
    rax = plt.axes([0.05, 0.9, 0.1, 0.08], frameon=True) # Changed second and fourth parameters
    check = CheckButtons(rax, ['Extend'], [False]) 

    def func(label):
        for line in random_lines:
            line.set_visible(not line.get_visible())
        
        # Check if the random lines are visible or not
        random_lines_visible = random_lines[0].get_visible()

        # Update the legend accordingly
        if random_lines_visible:
            # Add new legend handles for the extended trajectories
            light_gray_line = mlines.Line2D([], [], color='lightgray', marker='^', linestyle='-', markersize=4, label='Extended Predicted')
            dark_gray_line = mlines.Line2D([], [], color='darkgray', marker='o', linestyle='--', markersize=4, label='Extended Actual')
            ax.legend(handles=[red_line, black_line, light_gray_line, dark_gray_line])
        else:
            # Remove the additional legend handles
            ax.legend(handles=[red_line, black_line])
        
        plt.draw()

    check.on_clicked(func)

    plt.show()
    
def plot_losses(train_losses, val_losses, settings):
    # Extracting model_type from the settings
    model_type = settings.neural_net.model_type
    
    # Plotting the training and validation losses with updated aesthetics
    plt.figure(figsize=(10, 7))  # Set the figure size
    plt.plot(train_losses, color="darkred", linestyle="-", label="Training Loss")
    plt.plot(val_losses, color="steelblue", linestyle="--", label="Validation Loss")
    
    # Setting labels, title, and legend
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Losses for {model_type} Model")
    plt.legend()
    
    # Display the plot
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add a grid for better readability
    plt.tight_layout()  # Ensure that all elements fit within the figure space
    plt.show()
