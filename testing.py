import torch
from tqdm import tqdm
from settings import Settings

settings = Settings()

def run_emulator(model, test_loader):
    model = model.to(settings.execution.device).eval()  # Ensure model is on the correct device and set to evaluation mode
    predictions = []
    actual = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Running emulator'):
            inputs = inputs.to(settings.execution.device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())  # Move outputs to CPU before converting to numpy
            actual.append(targets.cpu().numpy())  # Move targets to CPU before converting to numpy
            
    return predictions, actual
