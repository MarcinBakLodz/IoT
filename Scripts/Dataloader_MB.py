import csv
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Step 1: Create a Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, file_path:str):
        self.sample_names_array:np.ndarray = self.read_csv_file(file_path)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label
    
    def read_csv_file(self, file_path:str)-> np.ndarray:
        try:
            with open(file_path, newline='', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter=';')  # Change delimiter if needed
                headers = next(reader) 
                reader_list = [row for row in reader] 
                reader_array = np.array(reader_list)
                return reader_array
        except Exception as e:
            print(f"Błąd podczas odczytu pliku: {e}")

    def shufle_samples(self, data: np.ndarray, train_size=0.7, val_size=0.15, random_state=0):
        """
        Shuffles and splits the given NumPy array into training, validation, and test sets.
        
        :param data: NumPy array containing data
        :param train_size: Proportion of data for training
        :param val_size: Proportion of data for validation
        :param test_size: Proportion of data for testing
        :param random_state: Random seed for reproducibility
        :return: train, val, test NumPy arrays
        """
        assert train_size - val_size < 1, "Train and validation size must be smaller or equal to 1"
        
        test_size = 1 - train_size - val_size 
        if random_state == 0:
            random_state = random.randint(0, 100)

        np.random.seed(random_state)
        np.random.shuffle(data)

        train_data, temp_data = train_test_split(data, test_size=(val_size + test_size), random_state=random_state)
        val_data, test_data = train_test_split(temp_data, test_size=(test_size / (val_size + test_size)), random_state=random_state)

        return train_data, val_data, test_data
    

if __name__ == "__main__":
    dataset = CustomDataset("Data\\opis_przejsc.csv")
    print(dataset.sample_names_array)
    print("-------------------------------------")
    train, val, test = dataset.shufle_samples(dataset.sample_names_array, random_state=42)
    print(train)
    