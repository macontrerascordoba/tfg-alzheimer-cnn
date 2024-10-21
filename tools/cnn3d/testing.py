import os
import torch

import torch.nn as nn
import pandas as pd
import numpy as np
import nibabel as nib

from torch.utils.data import DataLoader, Dataset
from tools import metrics
from pathlib import Path



class TestDatasetFromNii(Dataset):
    def __init__(self, csv_path):
        self.data_info = pd.read_csv(csv_path, header=None)
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        self.data_len = len(self.data_info.index)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        single_image_path = self.image_arr[index]
        single_image_nii = nib.load(single_image_path)
        single_image_arrary = single_image_nii.get_fdata()
        single_image_arrary = single_image_arrary.astype(np.float32)
        single_image_arrary = np.expand_dims(single_image_arrary, axis=0)
        
        img_as_tensor = torch.from_numpy(single_image_arrary)
        single_image_label = self.label_arr[index]
        label_as_tensor = torch.tensor(single_image_label, dtype=torch.long)
        return (img_as_tensor, label_as_tensor)



def generalization_test(model, device, seed, batch_size, file_diferentiator, results_path):

    # Test phase
    print("\nTesting the model...\n")

    test_csv = f'data/test_labels_{seed}.csv'

    test_dataset = TestDatasetFromNii(test_csv)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss().to(device)

    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Store labels and predictions for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_loss /= len(test_loader)
    test_accuracy = 100 * test_correct / test_total

    # Calculate confusion matrix
    cms = metrics.confusion_matrix(all_labels, all_predictions)

    # Calculate accuracy
    accuracy_list = metrics.calculate_accuracy(cms)

    # Calculate Precision and Recall
    precision_list, recall_list = metrics.calculate_precision_recall(cms)

    # Calculate F1 Score
    F1_list = metrics.calculate_f1(precision_list, recall_list)

    base_path = os.path.join(results_path, 'testing')
    Path(base_path).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(base_path, f'cms-{file_diferentiator}.txt'), 'w') as cms_file:

        print(f'\nConfusion Matrices:\n')
        
        for label, cm in cms.items():
            print(f'Class {label}:')
            print(f'{cm}\n')

            cms_file.write(f'Class {label}:\n{cm}\n\n')




    with open(os.path.join(base_path, f'by_class-{file_diferentiator}.csv'), 'w') as clases_file:

        clases_file.write('Class,Accuracy,Precision,Recall,F1 Score')

        print(f'Test Loss: {test_loss} | Test Global Accuracy: {test_accuracy}%\n')

        sum_acc = 0
        print(f'\nAccuracy by Class:\n')
        for label, accuracy in enumerate(accuracy_list):
            print(f'Class {label} Accuracy: {accuracy}%')
            sum_acc += accuracy

        print(f'\nMean Accuracy: {sum_acc / len(accuracy_list)}%\n')

        sum_prec = 0
        print(f'\nPrecision by Class:\n')
        for label, precision in enumerate(precision_list):
            print(f'Class {label} Precision: {precision}')
            sum_prec += precision

        print(f'\nMean Precision: {sum_prec / len(precision_list)}\n')

        sum_rec = 0
        print(f'\nRecall by Class:\n')
        for label, recall in enumerate(recall_list):
            print(f'Class {label} Recall: {recall}')
            sum_rec += recall

        print(f'\nMean Recall: {sum_rec / len(recall_list)}\n')

        sum_f1 = 0
        print(f'\nF1 Score by Class:\n')
        for label, F1 in enumerate(F1_list):
            print(f'Class {label} F1 Score: {F1}')
            sum_f1 += F1

        print(f'\nMean F1 Score: {sum_f1 / len(F1_list)}\n')

        for label in range(len(accuracy_list)):
            clases_file.write(f'\n{label},{accuracy_list[label]},{precision_list[label]},{recall_list[label]},{F1_list[label]}')



    with open(os.path.join(base_path, f'averages-{file_diferentiator}.csv'), 'w') as averages_file:

        averages_file.write('Accuracy,Precision,Recall,F1 Score')
        averages_file.write(f'\n{sum_acc / len(accuracy_list)},{sum_prec / len(precision_list)},{sum_rec / len(recall_list)},{sum_f1 / len(F1_list)}')


