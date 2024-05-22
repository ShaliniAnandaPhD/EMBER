import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from model_architecture import DocumentRetrievalModel, InstructionGenerationModel, ScoringModel
from data_preprocessing import WildfireDataset
from evaluation import evaluate_model
from utils import save_model, load_model, log_metrics, plot_metrics
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(model, train_loader, val_loader, epochs, optimizer, scheduler, device, model_save_path):
    """
    Train a model on the given data loaders.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): The data loader for training data.
        val_loader (DataLoader): The data loader for validation data.
        epochs (int): The number of epochs to train for.
        optimizer (optim.Optimizer): The optimizer to use for training.
        scheduler (lr_scheduler._LRScheduler): The learning rate scheduler.
        device (torch.device): The device to use for training (e.g., 'cuda' or 'cpu').
        model_save_path (str): The path to save the best model checkpoint.

    Returns:
        None
    """
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    try:
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            val_loss, val_metrics = evaluate_model(model, val_loader, device)
            val_losses.append(val_loss)

            log_metrics(epoch, train_loss, val_loss, val_metrics)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, model_save_path)

    except RuntimeError as e:
        logging.error(f"Runtime error occurred during training: {str(e)}")
        logging.error("Possible solution: Adjust the batch size or model architecture to fit the available memory.")

    except ValueError as e:
        logging.error(f"Value error occurred during training: {str(e)}")
        logging.error("Possible solution: Check the data preprocessing and ensure the data is in the expected format.")

    except KeyboardInterrupt:
        logging.info("Training interrupted by the user.")
        logging.info("Saving the current model state...")
        save_model(model, model_save_path)

    finally:
        plot_metrics(train_losses, val_losses)

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load preprocessed data
    train_dataset = WildfireDataset('path/to/preprocessed/train_data.pt')
    val_dataset = WildfireDataset('path/to/preprocessed/val_data.pt')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Initialize model architectures
    document_retrieval_model = DocumentRetrievalModel('bert-base-uncased', 768, 2).to(device)
    instruction_generation_model = InstructionGenerationModel('gpt2').to(device)
    scoring_model = ScoringModel('t5-base').to(device)

    # Define optimizer and learning rate scheduler
    optimizer = optim.AdamW(document_retrieval_model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * 10
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Train the models
    train_model(document_retrieval_model, train_loader, val_loader, epochs=10, optimizer=optimizer, scheduler=scheduler, device=device, model_save_path='path/to/save/document_retrieval_model.pt')
    train_model(instruction_generation_model, train_loader, val_loader, epochs=10, optimizer=optimizer, scheduler=scheduler, device=device, model_save_path='path/to/save/instruction_generation_model.pt')
    train_model(scoring_model, train_loader, val_loader, epochs=10, optimizer=optimizer, scheduler=scheduler, device=device, model_save_path='path/to/save/scoring_model.pt')

if __name__ == '__main__':
    main()