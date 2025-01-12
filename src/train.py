import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import MultimodalModel
from utils import save_model, load_model
from data_loader import MultimodalDataset  # Assuming a data_loader module exists
from text_processing import prepare_text_data
from audio_processing import process_audio
from video_processing import extract_frames

def train_model(train_data, val_data, epochs=10, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MultimodalModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            text, audio, video, labels = batch
            text, audio, video, labels = text.to(device), audio.to(device), video.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(text, audio, video)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

        # Validation step can be added here

    save_model(model, 'multimodal_sentiment_model.pth')

if __name__ == "__main__":
    train_data = MultimodalDataset('data/processed/train')  # Adjust path as necessary
    val_data = MultimodalDataset('data/processed/val')      # Adjust path as necessary
    train_model(train_data, val_data)