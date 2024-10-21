import click
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from pathlib import Path


def get_subtitle_metrics_from_filename(filename):

    seed = int(filename.split('-')[1].split('_')[-1])

    hyperparams = filename.split('-')[-1].split('_')
    model = hyperparams[0]
    folds = hyperparams[1]
    patience = hyperparams[2]
    lrate = '.'.join(hyperparams[3].split('.')[0:2])
    
    subtitle = f'Seed: {seed}, Model {model}: Folds = {folds}, Patience = {patience}, L. Rate = {float(lrate):.0e}'

    return seed, model, folds, patience, lrate, subtitle


def parse_cm_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    cm_matrices = {}
    class_idx = None
    matrix_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith("Class"):
            if class_idx is not None:
                # Save the previous class's confusion matrix
                cm_matrices[class_idx] = np.array(matrix_lines, dtype=int)
                matrix_lines = []
            class_idx = int(line.split()[1][:-1])  # Extract class index
        elif line.startswith("["):
            # Collect lines of the matrix
            matrix_lines.append(list(map(int, line.strip("[]").split())))
    
    if class_idx is not None:
        # Save the last class's confusion matrix
        cm_matrices[class_idx] = np.array(matrix_lines, dtype=int)

    return cm_matrices


def create_cm_heatmap(ax, cm, class_label):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, 
                xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])

    # Set the title based on the class label
    if class_label == 0:
        class_label_name = 'AD'
    elif class_label == 1:
        class_label_name = 'MCI'
    elif class_label == 2:
        class_label_name = 'CD'
    ax.set_title(f'Class {class_label_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    

def create_general_cm(cm_matrices, model_cms_path, seed, model, folds, patience, lrate, subtitle):
    # Initialize the 3x3 confusion matrix
    general_cm = np.zeros((3, 3), dtype=int)

    # Fill in the confusion matrix based on the individual class matrices

    # Class 0
    general_cm[0, 0] = cm_matrices[0][1, 1]  # True Positives for Class 0
    general_cm[0, 1] = cm_matrices[1][1, 0]  # False Positives for Class 1, from Class 0 perspective
    general_cm[0, 2] = cm_matrices[2][1, 0]  # False Positives for Class 2, from Class 0 perspective

    # Class 1
    general_cm[1, 1] = cm_matrices[1][1, 1]  # True Positives for Class 1
    general_cm[1, 0] = cm_matrices[0][1, 0]  # False Positives for Class 0, from Class 1 perspective
    general_cm[1, 2] = cm_matrices[2][0, 1]  # False Positives for Class 2, from Class 1 perspective

    # Class 2
    general_cm[2, 2] = cm_matrices[2][0, 0]  # True Positives for Class 2
    general_cm[2, 0] = cm_matrices[0][0, 1]  # False Positives for Class 0, from Class 2 perspective
    general_cm[2, 1] = cm_matrices[1][0, 1]  # False Positives for Class 1, from Class 2 perspective

    class_labels = ['Class AD', 'Class MCI', 'Class CN']

    plt.figure(figsize=(8, 6))
    sns.heatmap(general_cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix ({subtitle})')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Add some padding for the title

    best_save_folder = os.path.join(model_cms_path, f'general_cm-{seed}-{model}_{folds}_{patience}_{lrate}.pdf')
    plt.savefig(best_save_folder)
    plt.close()


def generate_graphs_folds(model, results_path, graphs_base_path):

    pattern = fr'folds-(\d+)_(\d+)-{model}_(\d+)_(\d+)_([\d.]+)\.csv'
    files = [f for f in os.listdir(os.path.join(results_path, 'training')) if re.search(pattern, f)]
    
    model_folds_path = os.path.join(graphs_base_path, f'model{model}', 'folds')
    Path(model_folds_path).mkdir(parents=True, exist_ok=True)

    # Metrics list
    metrics = ['Validation Loss', 'Global Accuracy', 'Mean Accuracy', 'Mean Precision', 'Mean Recall', 'Mean F1 Score']

    for file in files:
        
        # Load the CSV file
        df = pd.read_csv(os.path.join(results_path, 'training', file))
        
        seed, model, folds, patience, lrate, subtitle = get_subtitle_metrics_from_filename(file)
            
        # Create a plot for each metric
        for metric in metrics:
            
            title = f'{metric} ({subtitle})'
            x_label = 'Epoch'
            y_label = metric
            base_folder = os.path.join(model_folds_path, metric.lower().replace(" ", "_"))
            folder_name = f'{metric.lower().replace(" ", "_")}-{model}_{folds}_{patience}_{lrate}'
            save_folder = os.path.join(base_folder, folder_name)
            Path(save_folder).mkdir(parents=True, exist_ok=True)

            # Trace a line for each fold
            for fold in df['Fold'].unique():
                fold_data = df[df['Fold'] == fold]
                fold_data = fold_data[fold_data['Epoch'] != 1001]
                plt.plot(fold_data['Epoch'], fold_data[metric], label=f'Fold {fold}')
            
                plt.title(title)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.legend()

                # Guardar la gráfica en un archivo PNG
                save_filename = os.path.join(save_folder, f'Fold{fold}.pdf')
                plt.savefig(save_filename)
                plt.close()

        # Define the grid dimensions
        nrows, ncols = 2, 3

        # Create subplots: a 2x3 grid
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))

        # Flatten the axes array to easily iterate over it
        axes = axes.flatten()
        
        fig.suptitle(subtitle, fontsize=16)
        
        for idx, metric in enumerate(metrics):

            title = f'Best {metric}'
            x_label = 'Fold'
            y_label = metric

            best_value = df[(df['Epoch'] == 1001)]
            
            # Plot on the corresponding subplot
            axes[idx].plot(df['Fold'].unique(), best_value[metric])
            axes[idx].set_title(title)
            axes[idx].set_xlabel(x_label)
            axes[idx].set_ylabel(y_label)

        # Hide any unused subplots if there are fewer metrics than subplots
        for i in range(len(metrics), nrows * ncols):
            fig.delaxes(axes[i])

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Add some padding for the title


        best_save_folder = os.path.join(model_folds_path, f'best_metrics-{seed}-{model}_{folds}_{patience}_{lrate}.pdf')
        plt.savefig(best_save_folder)
        plt.close()


def generate_graphs_averages(model, mode, results_path, graphs_base_path):

    pattern = fr'averages-(\d+)_(\d+)-{model}_(\d+)_(\d+)_([\d.]+)\.csv'
    files = [f for f in os.listdir(os.path.join(results_path, mode)) if re.search(pattern, f)]
    
    model_averages_path = os.path.join(graphs_base_path, f'model{model}', 'averages')
    Path(model_averages_path).mkdir(parents=True, exist_ok=True)

    for file in files:

        df = pd.read_csv(os.path.join(results_path, mode, file))
            
        seed, model, folds, patience, lrate, subtitle = get_subtitle_metrics_from_filename(file)

        columns = list(df.columns)
        data = df.iloc[0].tolist()

        if mode == 'training':

            formatted_data = [
                round(data[0], 4),  # Validation Loss
                f"{round(data[1], 4)}%",  # Global Accuracy with percentage
                f"{round(data[2], 4)}%",  # Mean Accuracy with percentage
                round(data[3], 4),  # Mean Precision
                round(data[4], 4),  # Mean Recall
                round(data[5], 4)   # Mean F1 Score
            ]

        else:

            formatted_data = [
                f"{round(data[0], 4)}%",  # Accuracy
                round(data[1], 4),  # Precision
                round(data[2], 4),  # Recall
                round(data[3], 4)   # F1 Score
            ]

        fig, ax = plt.subplots(figsize=(8, 3))
        
        fig.suptitle(subtitle, fontsize=16)

        ax.axis('tight')
        ax.axis('off')

        table_data = [formatted_data]
        table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Add some padding for the title

        best_save_folder = os.path.join(model_averages_path, f'averages-{seed}-{model}_{folds}_{patience}_{lrate}.pdf')
        plt.savefig(best_save_folder)
        plt.close()


def generate_graphs_by_class(model, mode, results_path, graphs_base_path):
    
    pattern = fr'by_class-(\d+)_(\d+)-{model}_(\d+)_(\d+)_([\d.]+)\.csv'
    files = [f for f in os.listdir(os.path.join(results_path, mode)) if re.search(pattern, f)]
    
    model_by_class_path = os.path.join(graphs_base_path, f'model{model}', 'by_class')
    Path(model_by_class_path).mkdir(parents=True, exist_ok=True)

    for file in files:

        # Read the CSV file
        df = pd.read_csv(os.path.join(results_path, mode, file))
            
        seed, model, folds, patience, lrate, subtitle = get_subtitle_metrics_from_filename(file)

        # Extract the data
        columns = list(df.columns)
        data = df.values.tolist()

        # Format the data
        formatted_data = []
        for row in data:
            formatted_row = [
                int(row[0]),  # Class (as integer)
                f"{round(row[1], 2)}%",  # Accuracy with percentage
                round(row[2], 2),  # Precision
                round(row[3], 2),  # Recall
                round(row[4], 2)   # F1 Score
            ]
            formatted_data.append(formatted_row)

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 4))  # Adjust the figure size as needed
        
        fig.suptitle(subtitle, fontsize=16)

        ax.axis('tight')
        ax.axis('off')

        # Create the table
        table = ax.table(cellText=formatted_data, colLabels=columns, cellLoc='center', loc='center')

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Add some padding for the title

        best_save_folder = os.path.join(model_by_class_path, f'by_class-{seed}-{model}_{folds}_{patience}_{lrate}.pdf')
        plt.savefig(best_save_folder)
        plt.close()


def generate_cms(model, mode, results_path, graphs_base_path):
    
    pattern = fr'cms-(\d+)_(\d+)-{model}_(\d+)_(\d+)_([\d.]+)\.txt'
    files = [f for f in os.listdir(os.path.join(results_path, mode)) if re.search(pattern, f)]
    
    model_cms_path = os.path.join(graphs_base_path, f'model{model}', 'cms')
    Path(model_cms_path).mkdir(parents=True, exist_ok=True)

    for file in files:

        # Parse the confusion matrix file
        cm_matrices = parse_cm_file(os.path.join(results_path, mode, file))
            
        seed, model, folds, patience, lrate, subtitle = get_subtitle_metrics_from_filename(file)

        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))  # Adjust size as needed
        axs = axs.flatten()  # Flatten the array for easy indexing
        
        fig.suptitle(subtitle, fontsize=16)

        # Plot each class confusion matrix
        for i, (class_label, cm) in enumerate(cm_matrices.items()):
            create_cm_heatmap(axs[i], cm, class_label)

        # Hide any unused subplots if there are fewer metrics than subplots
        fig.delaxes(axs[3])

        # Adjust layout and save the figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Add some padding for the title

        best_save_folder = os.path.join(model_cms_path, f'cms-{seed}-{model}_{folds}_{patience}_{lrate}.pdf')
        plt.savefig(best_save_folder)
        plt.close()

        create_general_cm(cm_matrices, model_cms_path, seed, model, folds, patience, lrate, subtitle)



@click.command()
@click.option('--results_path', '-r', default='results', required=False, help='Path where the results are stored')
def main(results_path):

    ###################################
    # CHECKIN IF PROVIDED PATH EXISTS #
    ###################################
    if not os.path.exists(results_path):
        print(f'The specified path "{results_path}" does not exists')
        exit()


    ###################
    # TRAINING GRAPHS #
    ###################

    print('Generating Training Grahs...')

    # Training graphs base folder
    mode = 'training'
    graphs_base_path = os.path.join(results_path, 'graphs', mode)

    for model in range(1, 4):

        print(f'|\n|\tGenerating Model {model} Grahs...')

        print('|\t|\n|\t|\tGenerating Folds Grahs...')
        generate_graphs_folds(model, results_path, graphs_base_path)
        print('|\t|\t|\n|\t|\tFolds Grahs Generated')


        print('|\t|\n|\t|\tGenerating Averages Grahs...')
        generate_graphs_averages(model, mode, results_path, graphs_base_path)
        print('|\t|\t|\n|\t|\tAverages Grahs Generated')


        print(f'|\t|\n|\tModel {model} Grahs Generated')

    print('|\nTraining Grahs Generated')


    ##################
    # TESTING GRAPHS #
    ##################

    print('\nGenerating Testing Grahs...')

    # Training graphs base folder
    mode = 'testing'
    graphs_base_path = os.path.join(results_path, 'graphs', mode)

    for model in range(1, 4):

        print(f'|\n|\tGenerating Model {model} Grahs...')


        print('|\t|\n|\t|\tGenerating Averages Grahs...')
        generate_graphs_averages(model, mode, results_path, graphs_base_path)
        print('|\t|\t|\n|\t|\tAverages Grahs Generated')
        
        print('|\t|\n|\t|\tGenerating Folds Grahs...')
        generate_graphs_by_class(model, mode, results_path, graphs_base_path)
        print('|\t|\t|\n|\t|\tFolds Grahs Generated')
        
        print('|\t|\n|\t|\tGenerating CMs...')
        generate_cms(model, mode, results_path, graphs_base_path)
        print('|\t|\t|\n|\t|\tCMs Generated')

        print(f'|\t|\n|\tModel {model} Grahs Generated')


    print('|\nTraining Grahs Generated')


    print("\n\nGráficas creadas y guardadas exitosamente.")




if __name__ == "__main__":
    main()