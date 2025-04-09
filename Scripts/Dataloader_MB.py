import csv
import torch
import numpy as np
import random
import os
import enum
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class DataLoaderType(enum.Enum):
    POCKET = 1,
    WRIST = 2,
    POCKET_AND_WRIST = 3,
    POCKET_OR_WRIST = 4

class CustomDataset(Dataset):
    def __init__(self, description_file_path:str, data_directory_path:str, random_state:int = 0, mode:DataLoaderType = DataLoaderType.POCKET_AND_WRIST, debug:bool = True):
        self.files_to_read_list:list = ["acce", "game_rv", "gyro"]#["acce", "game_rv", "gyro", "gyro_bias", "linacce", "magnet", "magnet_bias", "pressure", "rv"]
        self.debug = debug
        
        if random_state == 0: random_state = random.randint(0, 100)
        self.sample_names_array:np.ndarray = self.read_description_file(description_file_path)
        self.train_file_name, self.val_file_name, self.test_file_name = self.shufle_and_split_samples(self.sample_names_array, random_state = random_state)
        
        self.train_sensor_reads = self.read_all_samples(self.train_file_name, data_directory_path, mode, "train")
        self.val_sensor_reads = self.read_all_samples(self.val_file_name, data_directory_path, mode, "val")
        self.test_sensor_reads = self.read_all_samples(self.test_file_name, data_directory_path, mode, "test")
        
    def read_all_samples(self, samples_array:np.ndarray, basic_directory:str, mode:DataLoaderType = DataLoaderType.POCKET_AND_WRIST, name:str = "default name")-> np.ndarray:
        result = []
        choosed_by_mode_samples_array = np.array([])
        if mode == DataLoaderType.POCKET:
            choosed_by_mode_samples_array = samples_array[:, [1, 3, 4, 5]]
            for sample, label, start, stop in choosed_by_mode_samples_array:
                sample_path = os.path.join(basic_directory, "Telefon_na_rece", sample)
                if self.debug: print(sample_path, " ", label) 
                all_sensors_read = self.read_all_sensors(sample_path, int(start), int(stop))
                if all_sensors_read.numel() > 0:  # Make sure it's not empty
                    result.append(all_sensors_read)
            print(mode)
        elif mode == DataLoaderType.WRIST:
            choosed_by_mode_samples_array = samples_array[:, [1, 3, 4, 5]]
            for sample, label, start, stop in choosed_by_mode_samples_array:
                sample_path = os.path.join(basic_directory, "Telefon_na_rece", sample)
                if self.debug: print(sample_path, " ", label) 
                all_sensors_read = self.read_all_sensors(sample_path, int(start), int(stop))
                if all_sensors_read.numel() > 0:  # Make sure it's not empty
                    result.append(all_sensors_read)
            print(mode)
        elif mode == DataLoaderType.POCKET_AND_WRIST or mode == DataLoaderType.POCKET_OR_WRIST:
            choosed_by_mode_samples_array = samples_array[:, 1:6]
            print(choosed_by_mode_samples_array)
            print(mode)
            
        if result:
            print(f"[{name}] Samples number: {len(result)}")
        else:
            result = torch.empty(0)
            print("[{name}] No valid data found across sessions.")
        
        return result
            
            
    def read_all_sensors(self, basic_path:str, bias_head:int, bias_tail:int):
        all_sensor_data = []

        for sensor in self.files_to_read_list:
            file_path = os.path.join(basic_path, sensor + ".txt")
            if self.debug: print(f"Reading file: {file_path}")

            try:
                raw_lines = []
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.startswith('#'):
                            continue  # Skip metadata/header lines
                        raw_lines.append(line.strip())

                # Apply head/tail bias
                raw_lines = raw_lines[bias_head:]
                if bias_tail > 0:
                    raw_lines = raw_lines[:-bias_tail]

                sensor_data = []
                for line in raw_lines:
                    parts = line.split()
                    if len(parts) < 2:
                        if self.debug:print(f"Skipping malformed line in {file_path}: {line}")
                        continue
                    try:
                        values = list(map(float, parts[1:]))  # Skip timestamp
                        sensor_data.append(values)
                    except ValueError as ve:
                        print(f"Skipping non-numeric line in {file_path}: {line} - {ve}")
                        continue

                all_sensor_data.append(sensor_data)

            except FileNotFoundError:
                print(f"[ERROR] File not found: {file_path}")
                return torch.empty(0)
            except Exception as e:
                print(f"[ERROR] Failed to read {file_path}: {e}")
                return torch.empty(0)

        # Combine sensor rows across files
        try:
            combined_rows = [
                sum(row_sets, [])  # Flatten values from all sensors for the same row
                for row_sets in zip(*all_sensor_data)
            ]
        except Exception as e:
            print(f"[ERROR] Failed to combine sensor data: {e}")
            return torch.empty(0)

        tensor = torch.tensor(combined_rows)
        print(f"Final tensor shape: {tensor.shape}")
        return tensor
            
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label
    
    def read_description_file(self, file_path:str)-> np.ndarray:
        try:
            with open(file_path, newline='', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter=';')  # Change delimiter if needed
                headers = next(reader) 
                reader_list = [row for row in reader] 
                reader_array = np.array(reader_list)
                return reader_array
        except Exception as e:
            print(f"Błąd podczas odczytu pliku: {e}")

    def shufle_and_split_samples(self, data: np.ndarray, train_size=0.7, val_size=0.15, random_state=0):
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
    dataset = CustomDataset("Data\\opis_przejsc.csv", "C:\\Users\\Marcin\\Desktop\\Studia\\IoT\\Data", random_state = 42, mode = DataLoaderType.POCKET, debug=False)
    print("-------------------------------------")

    