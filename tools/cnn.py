#!/home/i92cocom/miniconda3/envs/tfg/bin/python3

import click
import torch
import os

import pandas as pd

from datetime import datetime
from sklearn.model_selection import train_test_split

# Self-Created Libreries
from cnn3d.training import train_network
from cnn3d.loading import load_model
from cnn3d.testing import generalization_test
from cnn3d.predicting import predict


def split_labels(data_path, seed):

    # Load the base CSV file
    labels_df = pd.read_csv(data_path, header=None)

    # Split the data into training (80%) and testing (20%)
    train_df, test_df = train_test_split(labels_df, test_size=0.2, random_state=seed, stratify=labels_df[1])

    base_folder = '/'.join(data_path.split('/')[:-1])

    train_labels_path = os.path.join(base_folder, f'train_labels_{seed}.csv')
    test_labels_path = os.path.join(base_folder, f'test_labels_{seed}.csv')

    # Check if the file exists
    if os.path.isfile(train_labels_path):
        try:
            # Delete the file
            os.remove(train_labels_path)
        except Exception as e:
            print(f"An error occurred while trying to delete the file: {e}")

    # Check if the file exists
    if os.path.isfile(test_labels_path):
        try:
            # Delete the file
            os.remove(test_labels_path)
        except Exception as e:
            print(f"An error occurred while trying to delete the file: {e}")

    # Save the resulting dataframes to CSV files
    train_df.to_csv(train_labels_path, index=False, header=False)
    test_df.to_csv(test_labels_path, index=False, header=False)


@click.command()
@click.option('--seed', '-s', default=1, required=False, help=u'Seed to be used.')
@click.option('--date', '-d', default=datetime.now().strftime("%y%m%d%H%M"), required=False, help=u'Date of execution.')
@click.option('--data_path', '-D', default='data/labels.csv', required=False, help="CSV with all the labels for the images combined")
@click.option('--train', '-T', is_flag=True, help="Training Flag")
@click.option('--selected_model', '-M', default=1, required=False, help="Model selected to use on training")
@click.option('--trained_model_path', '-m', default='cnn3d_model.pth', required=False, help="Pretained model to use in case of no training")
@click.option('--test', '-t', is_flag=True, help="Testing Flag")
@click.option('--predict_path', '-p', default=None, required=False, help="Path of the image to predict")
@click.option('--batch_size', '-b', default=4, required=False, help="Size of the batches")
@click.option('--num_folds', '-f', default=5, required=False, help="Number of folds")
@click.option('--patience', '-w', default=10, required=False, help="Number of epochs to wait for improvement before stopping")
@click.option('--learning_rate', '-l', default=0.0001, required=False, help="Learning rate")
@click.option('--results_path', '-r', default='results', required=False, help='Path where the results are stored')
@click.pass_context
def main(ctx, seed, date, data_path, train, selected_model, trained_model_path, test, predict_path, batch_size, num_folds, patience, learning_rate, results_path):

    if not train and not test and predict_path == None:
        click.echo(ctx.get_help())
        ctx.exit()

    file_diferentiator = f'{date}_{seed}-{selected_model}_{num_folds}_{patience}_{learning_rate}'

    print(f'Start Date: {datetime.now().strftime("[ %d/%m/%y  %H:%M ]")}')

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = None

    split_labels(data_path, seed)

    selected_model = int(selected_model)
    batch_size = int(batch_size)
    num_folds = int(num_folds)
    patience = int(patience)
    learning_rate = float(learning_rate)

    print(f'Start Training: {datetime.now().strftime("[ %d/%m/%y  %H:%M ]")}')
    if train:
        model = train_network(seed, date, device, selected_model, batch_size, num_folds, patience, learning_rate, file_diferentiator, results_path)
        print(f'End Training: {datetime.now().strftime("[ %d/%m/%y  %H:%M ]")}')

    if test:
        if not train:
            model = load_model(trained_model_path, device)
            
        generalization_test(model, device, seed, batch_size, file_diferentiator, results_path)

    if not predict_path == None:
        if not train and not test:
            model = load_model(trained_model_path, device)

        probabilities = predict(model, device, predict_path)
        # prediction_label = ''
        # if prediction == 0: #AD
        #     prediction_label = 'AD'
        # elif prediction == 1:
        #     prediction_label = 'MCI'
        # elif prediction == 2:
        #     prediction_label = 'CN'

        print(f"Prediction for {predict_path.split('/')[-1]} is:\n{probabilities}")

    print(f'End Date: {datetime.now().strftime("[ %d/%m/%y  %H:%M ]")}')


if __name__ == "__main__":
    main()
