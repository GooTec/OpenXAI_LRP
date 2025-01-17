# Utils
import os
import time
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
import openxai.experiment_utils as utils

# Models, Data, and Explainers
from openxai.model import LoadModel
from openxai.dataloader import ReturnLoaders
from openxai.explainer import Explainer

if __name__ == '__main__':
    # Parameters
    config = utils.load_config('experiment_config.json')
    methods, n_test_samples = config['methods'], config['n_test_samples']
    param_strs = {method: utils.construct_param_string(config['explainers'][method]) for method in methods}

    # Generate explanations
    start_time = time.time()
    for data_name in config['data_names']:
        for model_name in config['model_names']:

            # Make directory for explanations
            folder_name = f'explanations/{model_name}_{data_name}'
            utils.make_directory(folder_name)
            print(f"Data: {data_name}, Model: {model_name}")
            
            # Load data
            trainloader, testloader = ReturnLoaders(data_name, download=True, batch_size=n_test_samples)
            inputs, labels = next(iter(testloader))
            inputs, labels = inputs.float(), labels.long()

            # Load model
            model = LoadModel(data_name, model_name, pretrained=True)
            predictions = model(inputs).argmax(dim=-1)

            # Convert predictions and labels to tensors (if needed)
            predictions_tensor = predictions.clone()  # Ensure predictions are PyTorch tensors

            # Loop over explanation methods
            for method in methods:
                # Print and configure
                print(f'Computing explanations for {method} (elapsed time: {time.time() - start_time:.2f}s)')
                param_dict = utils.fill_param_dict(method, config['explainers'][method], inputs)

                # Compute explanations
                explainer = Explainer(method, model, param_dict)
                explanations = explainer.get_explanations(inputs, predictions_tensor).detach().numpy()

                # Save as .npy
                npy_filename = f'{folder_name}/{method}_{n_test_samples}{param_strs[method]}.npy'
                np.save(npy_filename, {'explanations': explanations, 'predictions': predictions.numpy(), 'labels': labels.numpy()})
                print(f"Explanations saved to {npy_filename}")

                # Save as .csv
                csv_filename = f'{folder_name}/{method}_{n_test_samples}{param_strs[method]}.csv'
                df = pd.DataFrame(explanations, columns=[f"feature_{i}" for i in range(explanations.shape[1])])
                df['prediction'] = predictions.numpy()
                df['label'] = labels.numpy()
                df.to_csv(csv_filename, index=False)
                print(f"DataFrame saved to {csv_filename}")
