import torch 
import torch.nn as nn
from tqdm import tqdm 
import time  

class ModelTraining:
    @staticmethod
    def model_train(model, train_data, val_data, epochs=30):
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

        for epoch in range(epochs):
            start_time = time.time()  

            model.train()
            train_loss = 0
            train_loader = tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs} [Train]")  

            for images, labels in train_loader:
                images = images.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_loader.set_postfix({'train_loss': train_loss / (train_loader.n + 1)})

            avg_train_loss = train_loss / len(train_data)  

            model.eval()
            val_loss = 0
            val_loader = tqdm(val_data, desc=f"Epoch {epoch+1}/{epochs} [Val]")  

            with torch.no_grad():
                for images, label in val_loader:
                    images = images.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, label)
                    val_loss += loss.item()
                    val_loader.set_postfix({'val_loss': val_loss / (val_loader.n + 1)})

            avg_val_loss = val_loss / len(val_data) 

            scheduler.step(avg_val_loss)

            epoch_duration = time.time() - start_time

            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Validation Loss: {avg_val_loss:.4f}')
            print(f'Time Taken: {epoch_duration:.2f} seconds')
            print('-' * 50)