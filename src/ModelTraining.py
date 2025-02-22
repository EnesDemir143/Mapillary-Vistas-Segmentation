import torch
import torch.nn as nn
from tqdm import tqdm
import time
import os

class ModelTraining:
    @staticmethod
    def save_checkpoint(model, optimizer, scheduler, epoch, filepath='checkpoint.pth'):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved at epoch {epoch+1} to {filepath}")

    @staticmethod
    def load_checkpoint(model, optimizer, scheduler, filepath='checkpoint.pth'):
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Checkpoint loaded from {filepath}, resuming from epoch {start_epoch}")
            return start_epoch
        else:
            print(f"No checkpoint found at {filepath}, starting from scratch")
            return 0

    @staticmethod
    def model_train(model, train_data, val_data, epochs=30, checkpoint_path='checkpoint.pth', resume=False):
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

        start_epoch = 0
        if resume:
            start_epoch = ModelTraining.load_checkpoint(model, optimizer, scheduler, checkpoint_path)

        for epoch in range(start_epoch, epochs):
            start_time = time.time()

            model.train()
            train_loss = 0
            train_loader = tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs} [Train]")

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

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
                for images, labels in val_loader: 
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
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

            if epoch + 1 == 5:
                ModelTraining.save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)
                print("Training paused at epoch 5. Resume later with resume=True.")
                break  