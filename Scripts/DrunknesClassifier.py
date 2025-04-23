from comet_ml import Experiment
from dotenv import load_dotenv
import os
import torch
import torch.nn as nn
from Dataloader_MB import CustomDataset, DataLoaderType, DataType
from torch.utils.data import DataLoader



load_dotenv()
api_key = os.getenv("COMET_API_KEY")



class DrunknesClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=6, dropout=0.3, bidirectional=False):
        super(DrunknesClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,  # channels
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x) 
        last_hidden = hn[-1]  # shape: [batch, hidden]
        logits = self.fc(last_hidden)
        return logits
    
    def fit(self, number_of_epochs: int, train_loader, val_loader):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)  # self, nie model

        for epoch in range(number_of_epochs):
            print(f"\nEpoch {epoch + 1}/{number_of_epochs}")
            self.train()  # przełączenie w tryb treningowy
            self.train_phase(train_loader, criterion, optimizer)

            self.eval()  # przełączenie w tryb walidacyjny
            self.val_phase(val_loader, criterion)

    def train_phase(self, loader, criterion, optimizer):
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in loader:
            labels = labels.long()
            optimizer.zero_grad()

            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total * 100
        print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    def val_phase(self, loader, criterion):
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                labels = labels.long()
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total * 100
        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
    # preds = torch.argmax(outputs, dim=1)  # shape [batch], dtype lon
    
if __name__ == "__main__":
    experiment = Experiment(
        api_key = os.getenv("COMET_API_KEY")
        project_name="drunkness-classifier",
        workspace="YOUR_WORKSPACE"
    )
    batch_size = 32
    seq_len = 200
    channels = 27

    x = torch.randn(batch_size, seq_len, channels)

    model = DrunknesClassifier(input_size=27)
    print(model(x).shape)  # shape: [batch, 6]
    
    dataset = CustomDataset(False, "Data\\opis_przejsc.csv", "C:\\Users\\Marcin\\Desktop\\Studia\\IoT\\Data", data_from_samples_ratio=3, data_lenght = 400 , random_state = 42, mode = DataLoaderType.POCKET, dataset_directory =r"C:\Users\Marcin\Desktop\Studia\IoT\Data\Tensory\POCKET20250413_114501", debug=False)
    dataset.set_datatype(DataType.TRAIN)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    dataset.set_datatype(DataType.VALIDATION)
    val_loader = DataLoader(dataset, batch_size=32)
    dataset.set_datatype(DataType.TEST)
    test_loader = DataLoader(dataset, batch_size=32)

    model.fit(1000, train_loader, val_loader)