# 🌿 MINT Emulator

The MINT Emulator is a package designed to emulate prevalence data. The current version offers a variety of neural network architectures including Feedforward Neural Networks, GRUs, LSTMs, and BiRNNs.

## 📚 Table of Contents
- [🌿 MINT Emulator](#-mint-emulator)
	- [📚 Table of Contents](#-table-of-contents)
	- [🚀 Main Execution ](#-main-execution-)
	- [📦 Datasets and Data Handling ](#-datasets-and-data-handling-)
		- [MintDataset](#mintdataset)
		- [Data Handler](#data-handler)
	- [🧠 Model Architectures ](#-model-architectures-)
		- [FFNN (Feedforward Neural Network)](#ffnn-feedforward-neural-network)
		- [GRU](#gru)
		- [LSTM](#lstm)
		- [BiRNN (Bidirectional RNN)](#birnn-bidirectional-rnn)
	- [🛠 Model Handling ](#-model-handling-)
	- [🏋️‍♂️ Training Module ](#️️-training-module-)
	- [🧪 Testing the Emulator ](#-testing-the-emulator-)
	- [🔗 Dependencies and Libraries ](#-dependencies-and-libraries-)
	- [💡 Hyperparameter Tuning ](#-hyperparameter-tuning-)
		- [Key Hyperparameters:](#key-hyperparameters)
	- [📊 Plotting and Visualization ](#-plotting-and-visualization-)
	- [⚙️ Settings ](#️-settings-)
	- [🔧 Utilities ](#-utilities-)
	- [🚧 Error Handling and Logging ](#-error-handling-and-logging-)
	- [💼 Use Cases and Examples ](#-use-cases-and-examples-)
	- [💌 Feedback and Contribution ](#-feedback-and-contribution-)

---

## 🚀 Main Execution <a name="main-execution"></a>
This is the main execution script for the package. It prepares the data, handles the model (training or loading pre-existing models), runs the emulator, and visualizes the results.

Key functions and procedures:
- `prepare_data`: Loads and prepares training, validation, and test data.
- `handle_model`: Manages model initialization, loading, and training.
- `run_emulator`: Executes the emulator to get predictions.
- Various plotting functions (from the `plotting` module) to visualize results.

---

## 📦 Datasets and Data Handling <a name="datasets-and-data-handling"></a>
This section contains the data representation and loading procedures.

### MintDataset
A custom PyTorch dataset class used to load data from a CSV file. Data is split into inputs and outputs based on settings configurations.

Key Methods:
- `__init__`: Initializes the dataset, loading data from a given input file.
- `__len__`: Returns the total number of entries.
- `__getitem__`: Retrieves a specific entry, splitting it into input and output.

### Data Handler
Key function:
- `prepare_data`: Prepares the dataset for training, validation, and testing. It splits the data based on predefined percentages.

---

## 🧠 Model Architectures <a name="model-architectures"></a>
This section contains various neural network architectures.

### FFNN (Feedforward Neural Network)
A simple FFNN with dropout and batch normalization.

### GRU
A Gated Recurrent Unit (GRU) with dropout and layer normalization.

### LSTM
An LSTM with dropout and layer normalization.

### BiRNN (Bidirectional RNN)
A Bidirectional RNN with dropout and layer normalization.

## 🛠 Model Handling <a name="model-handling"></a>
Handles model-related activities including initialization, loading, and training.

Key Functions:
- `initialize_model`: Sets up the model based on the settings configuration.
- `load_pretrained_model`: Loads a pretrained model if one exists.
- `train_and_save_model`: Handles training the model and saving it post-training.
- `handle_model`: Orchestrates the entire process of initializing, loading, and/or training the model.

---

## 🏋️‍♂️ Training Module <a name="training-module"></a>
Contains the training routine for the neural network models.

Key Function:
- `train_model`: Takes a model, criterion (loss function), optimizer, training data, and validation data as inputs. The function then trains the model and validates it at the end of each epoch. Learning rate adjustment is also incorporated.

---

## 🧪 Testing the Emulator <a name="testing-the-emulator"></a>
Provides functionalities to run the emulator and get predictions on test data.

Key Function:
- `run_emulator`: Takes the trained model and test data as inputs. Executes the emulator and retrieves predictions and actual values.

---

## 🔗 Dependencies and Libraries <a name="dependencies-and-libraries"></a>
This section lists external libraries and dependencies required by the MINT Emulator.

- `PyTorch`: For building and training the neural network models.
- `NumPy`: Used for numerical operations.
- `pandas`: For data manipulation.
- `matplotlib`: For visualization and plotting.
- `tqdm`: For progress bars during training.

---

## 💡 Hyperparameter Tuning <a name="hyperparameter-tuning"></a>
Adjusting hyperparameters is crucial to obtain optimal performance from the models. 

### Key Hyperparameters:
- `Learning Rate`: Determines the step size at each iteration while moving towards a minimum of the loss function.
- `Dropout Rate`: Prevents overfitting by randomly setting a fraction of the input units to 0 during training.
- `Batch Size`: Number of training examples used in one iteration.
- `Epochs`: Number of complete passes through the training dataset.

Hyperparameters can be adjusted in the `settings` module.

---
## 📊 Plotting and Visualization <a name="plotting-and-visualization-placeholder"></a>

The `plotting` module provides functionalities to visualize and compare the model's predictions with actual data:

1. `plot_mint_compare(predictions, actual, settings, num_samples=9)`: Displays scatter plots comparing actual versus predicted values for randomly chosen time series samples.
2. `plot_mint_time_series(predictions, actual, settings, num_samples=9)`: Visualizes the time series of the actual versus predicted values for random samples.
3. `plot_mint_avg_compare(predictions, actual, settings)`: Presents a scatter plot of average actual versus average predicted values.
4. `plot_mint_avg_time_series(predictions, actual, settings)`: Plots time series data of average actual versus average predicted values with options to display random trajectories.
5. `plot_losses(train_losses, val_losses)`: Visualizes the loss progression during training and validation across epochs.


<!-- --- -->

<!-- ## 📖 Documentation and Comments <a name="documentation-and-comments"></a>
Every function, method, and class within the MINT Emulator package is accompanied by detailed docstrings that follow the [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md). This makes it easy for developers and users to understand the purpose, parameters, and return values of each code section.
 -->

---

## ⚙️ Settings <a name="settings-placeholder"></a>

In the `settings` module, a variety of dataclass configurations are provided to manage different settings related to data, execution, neural networks, plotting, and learning rate scheduling:

1. `MINTData`: This handles settings for preprocessing the data, specifying the path to the MINT data file.
2. `Execution`: Manages settings related to execution like device (CPU or GPU), data source, number of workers, and the choice of using cached models.
3. `LrScheduler`: Configuration for learning rate scheduling including initial learning rate, step size, and gamma (decay factor).
4. `NeuralNet`: Contains neural network settings like epochs, batch size, dropout probability, data shuffling, number of workers for DataLoader, fraction of data for testing and validation, and input-output neuron sizes.
5. `Plotting`: Settings for plotting figures in different modes.
6. `Settings`: A consolidated class containing all the above configurations for easier access and management.

---

## 🔧 Utilities <a name="utilities-placeholder"></a>

The `utils` module consists of helper functions designed to aid in data and model management. These functions include:

1. `check_data_folder_exists(folder_path)`: Verifies the existence of a data folder and creates it if it doesn't exist.
2. `check_model_exists(model_type, source)`: Checks if a model of the specified type and source exists in the cache.
3. `select_model(settings)`: Based on the specified model type in the settings, this function returns an instance of the model (FFNN, GRU, LSTM, or BiRNN).
4. `attach_identifier(data)`: Attaches an identifier to data which is a list of numpy arrays.

---
## 🚧 Error Handling and Logging <a name="error-handling-and-logging"></a>
To ensure smooth operation, the MINT Emulator package incorporates error handling for common issues like:

- Missing data files
- Incompatible data formats
- Model mismatch issues (e.g., trying to load a GRU model for an LSTM setup)

In addition, a logging system is in place to provide detailed feedback during execution. Logs help in debugging and understanding the flow of the program.

---

## 💼 Use Cases and Examples <a name="use-cases-and-examples"></a>
For users unfamiliar with the MINT Emulator package, there are example scripts and use-cases provided to demonstrate:

- How to set up data
- How to initialize and train a model
- How to run predictions on new data
- How to visualize the results

Each example comes with detailed comments and explanations, making it easier for users to adapt and build upon these scripts for their own projects.

---



## 💌 Feedback and Contribution <a name="feedback-and-contribution"></a>
Feedback is crucial for the growth and improvement of the MINT Emulator. Users are encouraged to:

- Report bugs and issues
- Suggest new features or improvements
- Contribute to the codebase

Details for contributing and contact information can be found in the `CONTRIBUTING.md` and `README.md` files respectively.

---