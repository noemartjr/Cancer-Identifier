import subprocess
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def install_lib(python_lib: str) -> None:
    """
        Simple method for installing python library to local venv.

        :param python_lib: (str) python library to install
        :return: None
    """
    subprocess.run(['pip', 'install', python_lib])

def retrieve_csv_file(file_path: str):
    """
        Retrieves the csv file and returns the pandas dataframe

        :param file_path: (str) path to csv file
        :return: pandas dataframe
    """
    return pd.read_csv(file_path)

def remove_features(features) -> dict:
    """
        Creates a dictionary of copies of features, each with 0 or 1 features missing

        :param features: (pandas dataframe) features to use
        :return: (dict) {missing feature: pandas dataframe with feature missing}
    """
    return_dict = {'': features}
    for missing_feat_ind in features.columns:
        return_dict.update({missing_feat_ind: features.drop(missing_feat_ind, axis=1)})
    return return_dict

def get_folds(x, num_splits: int=5, shuffle: bool=True) -> list:
    """
        Returns a list of folds to use.

        :param x: (pandas dataframe) features to be used
        :param num_splits: (int) number of splits to make
        :param shuffle: (bool) shuffle the featers
        :return: (list) list of folds
    """
    kf = KFold(n_splits=num_splits, shuffle=shuffle)
    return list(kf.split(x))

class classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class predictor_model():
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, train_loader, num_epochs: int = 10) -> list:
        return_losses = []

        for epoch in range(num_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0

            for inputs, labels in train_loader:
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                # print(outputs)
                # print(labels)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            average_loss = running_loss / len(train_loader)
            return_losses.append(average_loss)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

        return return_losses

    def evaluate(self, val_loader) -> float:
        self.model.eval()  # Set the model to evaluation mode
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                predicted = self.model(inputs)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        accuracy = correct_predictions / total_samples
        print(f'Validation Accuracy: {accuracy * 100:.2f}%')

        return accuracy


def get_data(path: str = 'survey lung cancer.csv'):
    """
        Retrieves x and y data from csv file

        :param path: (str) relative path to csv file
        :return: x features, y labels
    """
    data = retrieve_csv_file(path)

    # print(data)

    y = data.pop('LUNG_CANCER')
    y = y.map({'YES': 1, 'NO': 0})

    x = data
    x['GENDER'] = x['GENDER'].map({'M': 0, 'F': 1})

    # print(x)
    # print(y)

    return x, y

def count_info(data) -> dict:
    """
        Counts the number of people there are in each classification

        :param data: (pandas dataframe) data to process
        :return: (dict) {feature: list of people in each classification}
    """
    return_data = {}

    for col_name in data.columns:
        col_vals = []
        vals = data[col_name]
        for unique_val in set(vals):
            col_vals.append(sum([x == unique_val for x in vals]))
        return_data[col_name] = col_vals

    return return_data

def create_bar_graph(data: dict) -> None:
    """
        Creates a bar graph using the data returned by count_info
        :param data: (dict) data returned by count_info
        :return: None
    """
    plt.figure(figsize=(10, 10))
    for i, data_vals in enumerate(data.values()):
        bottom = 0
        for ind_val in data_vals:
            plt.bar(i, ind_val, bottom=bottom)
            bottom += ind_val
    plt.xticks(list(range(len(data.keys()))), data.keys(), rotation='vertical')
    plt.subplots_adjust(bottom=0.25)
    plt.show()

if __name__ == '__main__':

    # pd.set_option('display.max_rows', None)  # Display all rows
    pd.set_option('display.max_columns', None)  # Display all columns
    pd.set_option('expand_frame_repr', False)  # Prevent the DataFrame from being truncated horizontally

    x, y = get_data()

    # create_bar_graph(count_info(retrieve_csv_file('survey lung cancer.csv')))

    for fold_num, (train, test) in enumerate(get_folds(x, shuffle=False)[:1]):
        print(f"Fold {fold_num}: ")
        X_train = torch.Tensor(x.iloc[train, :].values)
        y_train = torch.Tensor(y.iloc[train].values)
        y_train = y_train.view(y_train.shape[0], 1)
        X_test = torch.Tensor(x.iloc[test, :].values)
        y_test = torch.Tensor(y.iloc[test].values)
        y_test = y_test.view(y_test.shape[0], 1)

        # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_test, y_test)

        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Instantiate the model, define loss function, and optimizer
        model = classifier(input_size=x.shape[1], hidden_size=20, num_classes=1)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        pred_model = predictor_model(model, criterion, optimizer)

        # train model
        plt.plot(pred_model.train(train_loader, 50))
        plt.show()

        # validate model
        pred_model.evaluate(val_loader)
