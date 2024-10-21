import os
import torch
from cnn3d.models.cnn3d_own import Cnn3DOwn
from cnn3d.models.cnn3d_elassy import Cnn3DElAssy
from cnn3d.models.cnn3d_googlenet import Cnn3DGoogleNet

import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import nibabel as nib
import torchio as tio

from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path

from tools import metrics




# Custom dataloader
class DatasetFromNii(Dataset):      
    def __init__(self, csv_path, augment=False):
        self.data_info = pd.read_csv(csv_path, header=None)
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        self.data_len = len(self.data_info.index)
        self.augment = augment
        if self.augment:
            self.transforms = tio.Compose([
                tio.RandomFlip(axes=(0, 1, 2)),
                tio.RandomAffine(),
                tio.RandomNoise(),
            ])

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        single_image_path = self.image_arr[index]
        single_image_nii = nib.load(single_image_path)
        single_image_arrary = single_image_nii.get_fdata()
        single_image_arrary = single_image_arrary.astype(np.float32)
        single_image_arrary = np.expand_dims(single_image_arrary, axis=0)
        
        if self.augment:
            single_image_arrary = self.transforms(single_image_arrary)
        
        img_as_tensor = torch.from_numpy(single_image_arrary)
        single_image_label = self.label_arr[index]
        label_as_tensor = torch.tensor(single_image_label, dtype=torch.long)
        return (img_as_tensor, label_as_tensor)


def train_network(seed, date, device, selected_model, batch_size, num_folds, patience, learning_rate, file_diferentiator, results_path):

    # Parameters
    csv_path = f'data/train_labels_{seed}.csv'
    num_epochs = 1000  # Increased number of epochs for early stopping
    

    
    # Total memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    # Cached memory
    cached_memory = torch.cuda.memory_reserved(0)
    # Allocated memory
    allocated_memory = torch.cuda.memory_allocated(0)
    # Free memory
    free_memory = total_memory - (cached_memory + allocated_memory)

    print(f'Using device: {device} ( {free_memory / 1024**3:.2f} GB / {total_memory / 1024**3:.2f} GB )\n')


    print('Hiperparametros:\n')

    print(f'\tBatch Size: {batch_size}')
    print(f'\tLearning Rate: {learning_rate}')
    print(f'\tNumber of Epochs: {num_epochs}')
    print(f'\tNumber of Folds: {num_folds}')
    print(f'\tPatience: {patience}')
    if selected_model == 1:
        print(f'\tModel: Created by Student')
    elif selected_model == 2:
        print(f'\tModel: Created by El-Assy, A. M.')
    elif selected_model == 3:
        print(f'\tModel: GoogleNet')


    print('\n\n----------------------------------------------------\n\n')


    torch.manual_seed(seed)

    # Load dataset
    dataset = DatasetFromNii(csv_path, augment=True)
    dataset_size = len(dataset)
    labels = dataset.label_arr

    # Initialize K-Fold cross-validation
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    base_path = os.path.join(results_path, 'training')
    Path(base_path).mkdir(parents=True, exist_ok=True)

    # Cross-validation loop
    with open(os.path.join(base_path, f'folds-{file_diferentiator}.csv'), 'w') as folds_file:

        folds_file.write('Fold,Epoch,Validation Loss,Global Accuracy,Mean Accuracy,Mean Precision,Mean Recall,Mean F1 Score')
    
        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(dataset_size), labels)):
            print(f'Fold {fold + 1}/{num_folds}')
            
            # Subset the data for the current fold
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
            
            # Model, loss function, and optimizer
            if selected_model == 1:
                model = Cnn3DOwn().to(device)
            elif selected_model == 2:
                model = Cnn3DElAssy().to(device)
            elif selected_model == 3:
                model = Cnn3DGoogleNet().to(device)

            criterion = nn.CrossEntropyLoss().to(device)

            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
            # Instantiate the scheduler
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
            
            best_val_loss = float('inf')
            best_glob_accu = 0
            best_accu = 0
            best_prec = 0
            best_rec = 0
            best_f1 = 0
            patience_counter = 0
            best_model_state = None

            # Training loop
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                for i, data in enumerate(train_loader, 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    if i % 10 == 9:
                        print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10}')
                        running_loss = 0.0

                # Validation loop
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                all_labels = []
                all_predictions = []

                with torch.no_grad():
                    for data in val_loader:
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                        # Store labels and predictions for confusion matrix
                        all_labels.extend(labels.cpu().numpy())
                        all_predictions.extend(predicted.cpu().numpy())

                val_loss /= len(val_loader)
                global_accuracy = 100 * correct / total
                print(f'Epoch {epoch + 1}, Validation Loss: {val_loss}, Global Accuracy: {global_accuracy}%')

                # Calculate confusion matrix
                cms = metrics.confusion_matrix(all_labels, all_predictions)

                # Calculate accuracy
                accuracy_list = metrics.calculate_accuracy(cms)

                # Calculate Precision and Recall
                precision_list, recall_list = metrics.calculate_precision_recall(cms)

                # Calculate F1 Score
                F1_list = metrics.calculate_f1(precision_list, recall_list)

                accuracy = sum(accuracy_list) / len(accuracy_list)
                print(f'Mean Accuracy: {accuracy}%')

                precision = sum(precision_list) / len(precision_list)
                print(f'Mean Precision: {precision}')

                recall = sum(recall_list) / len(recall_list)
                print(f'Mean Recall: {recall}')

                f1_score = sum(F1_list) / len(F1_list)
                print(f'Mean F1 Score: {f1_score}\n')

                folds_file.write(f'\n{fold + 1},{epoch + 1},{val_loss},{global_accuracy},{accuracy},{precision},{recall},{f1_score}')

                # Step the scheduler based on validation loss
                scheduler.step(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_glob_accu = global_accuracy
                    best_accu = accuracy
                    best_prec = precision
                    best_rec = recall
                    best_f1 = f1_score
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'Early stopping at epoch {epoch + 1}')
                        break
        
            if best_model_state is not None:
            	model.load_state_dict(best_model_state)
            
            fold_results.append((best_val_loss, best_glob_accu, best_accu, best_prec, best_rec, best_f1))
            print(f'Fold {fold + 1}, Best Validation Loss: {best_val_loss}, Global Accuracy: {best_glob_accu}%')
            print(f'Mean Accuracy: {best_accu}%')
            print(f'Mean Precision: {best_prec}')
            print(f'Mean Recall: {best_rec}')
            print(f'Mean F1 Score: {best_f1}\n\n')

            folds_file.write(f'\n{fold + 1},{1001},{best_val_loss},{best_glob_accu},{best_accu},{best_prec},{best_rec},{best_f1}')

    with open(os.path.join(base_path, f'averages-{file_diferentiator}.csv'), 'w') as averages_file:

        averages_file.write('Validation Loss,Global Accuracy,Mean Accuracy,Mean Precision,Mean Recall,Mean F1 Score')

        # Average results across folds
        avg_val_loss = sum([result[0] for result in fold_results]) / num_folds
        avg_global_accuracy = sum([result[1] for result in fold_results]) / num_folds
        avg_mean_accuracy = sum([result[3] for result in fold_results]) / num_folds
        avg_mean_precision = sum([result[2] for result in fold_results]) / num_folds
        avg_mean_recall = sum([result[4] for result in fold_results]) / num_folds
        avg_mean_f1_score = sum([result[5] for result in fold_results]) / num_folds
        print(f'Average Validation Loss: {avg_val_loss}, Average Global Accuracy: {avg_global_accuracy}%')
        print(f'Average Mean Accuracy: {avg_mean_accuracy}%')
        print(f'Average Mean Precision: {avg_mean_precision}')
        print(f'Average Mean Recall: {avg_mean_recall}')
        print(f'Average Mean F1 Score: {avg_mean_f1_score}\n\n')

        averages_file.write(f'\n{avg_val_loss},{avg_global_accuracy},{avg_mean_accuracy},{avg_mean_precision},{avg_mean_recall},{avg_mean_f1_score}')

    # Save the model
    torch.save(model.state_dict(), f'{file_diferentiator}-cnn3d_model.pth')


    return model
