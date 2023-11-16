import torch
import numpy as np
from data_handler import prepare_data
from testing import run_emulator
from plotting import plot_mint_compare, plot_mint_time_series, plot_mint_avg_compare, plot_mint_avg_time_series
from model_handler import handle_model
from settings import Settings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(predictions, actual):
    """
    Calculates MAE, MSE, RMSE and R^2 for the given predictions and actual values.
    """
    # Flatten the predictions and actual values into one-dimensional arrays
    flattened_predictions = np.concatenate(predictions, axis=0)
    flattened_actual = np.concatenate(actual, axis=0)

    # Compute the average predicted and actual values
    avg_predicted = np.mean(flattened_predictions, axis=0)
    avg_actual = np.mean(flattened_actual, axis=0)

    # Compute the metrics on the average values
    mae = mean_absolute_error(avg_actual, avg_predicted)
    mse = mean_squared_error(avg_actual, avg_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(avg_actual, avg_predicted)
    
    return mae, mse, rmse, r2


if __name__ == "__main__":
    
    settings = Settings()

    train_loader, val_loader, test_loader, scaler = prepare_data(settings)

    # Handle model (loading, training, etc.)
    model = handle_model(train_loader, val_loader, settings)
    # Run the emulator
    predictions, actual = run_emulator(model, test_loader)
    
    mae, mse, rmse, r2 = calculate_metrics(predictions, actual)
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}")
    
    plot_mint_compare(predictions, actual, settings)
    plot_mint_time_series(predictions, actual, settings)
    plot_mint_avg_compare(predictions, actual, settings)
    plot_mint_avg_time_series(predictions, actual, settings)