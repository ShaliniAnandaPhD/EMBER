import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import WildfireDataset
from model_architecture import DocumentRetrievalModel, InstructionGenerationModel, ScoringModel
from evaluation_metrics import perplexity, bleu_score, rouge_score
from utils import load_model, generate_report, create_heatmap, create_confusion_matrix
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_document_retrieval(model, test_loader, device):
    """
    Evaluate the document retrieval model on the test set.

    Args:
        model (DocumentRetrievalModel): The trained document retrieval model.
        test_loader (DataLoader): The data loader for the test set.
        device (torch.device): The device to use for evaluation.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.tolist())
            true_labels.extend(labels.tolist())

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    return metrics

def evaluate_instruction_generation(model, test_loader, device):
    """
    Evaluate the instruction generation model on the test set.

    Args:
        model (InstructionGenerationModel): The trained instruction generation model.
        test_loader (DataLoader): The data loader for the test set.
        device (torch.device): The device to use for evaluation.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    model.eval()
    perplexity_scores = []
    bleu_scores = []
    rouge_scores = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            logits = outputs.logits

            # Calculate perplexity
            ppl = perplexity(logits, labels)
            perplexity_scores.append(ppl.item())

            # Generate instructions
            generated_instructions = model.generate(input_ids, max_length=100)

            # Calculate BLEU score
            bleu = bleu_score(generated_instructions, labels)
            bleu_scores.append(bleu)

            # Calculate ROUGE score
            rouge = rouge_score(generated_instructions, labels)
            rouge_scores.append(rouge)

    avg_perplexity = sum(perplexity_scores) / len(perplexity_scores)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge = sum(rouge_scores) / len(rouge_scores)

    metrics = {
        'perplexity': avg_perplexity,
        'bleu': avg_bleu,
        'rouge': avg_rouge
    }

    return metrics

def evaluate_scoring(model, test_loader, device):
    """
    Evaluate the scoring model on the test set.

    Args:
        model (ScoringModel): The trained scoring model.
        test_loader (DataLoader): The data loader for the test set.
        device (torch.device): The device to use for evaluation.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    model.eval()
    predictions = []
    true_scores = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['scores'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.tolist())
            true_scores.extend(scores.tolist())

    accuracy = accuracy_score(true_scores, predictions)
    precision = precision_score(true_scores, predictions, average='weighted')
    recall = recall_score(true_scores, predictions, average='weighted')
    f1 = f1_score(true_scores, predictions, average='weighted')

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    return metrics

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data
    test_dataset = WildfireDataset('path/to/preprocessed/test_data.pt')
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Load trained models
    document_retrieval_model = load_model('path/to/trained/document_retrieval_model.pt', DocumentRetrievalModel).to(device)
    instruction_generation_model = load_model('path/to/trained/instruction_generation_model.pt', InstructionGenerationModel).to(device)
    scoring_model = load_model('path/to/trained/scoring_model.pt', ScoringModel).to(device)

    try:
        # Evaluate document retrieval model
        logging.info("Evaluating document retrieval model...")
        document_retrieval_metrics = evaluate_document_retrieval(document_retrieval_model, test_loader, device)
        logging.info(f"Document Retrieval Metrics: {document_retrieval_metrics}")

        # Evaluate instruction generation model
        logging.info("Evaluating instruction generation model...")
        instruction_generation_metrics = evaluate_instruction_generation(instruction_generation_model, test_loader, device)
        logging.info(f"Instruction Generation Metrics: {instruction_generation_metrics}")

        # Evaluate scoring model
        logging.info("Evaluating scoring model...")
        scoring_metrics = evaluate_scoring(scoring_model, test_loader, device)
        logging.info(f"Scoring Metrics: {scoring_metrics}")

        # Generate evaluation report
        report = generate_report(document_retrieval_metrics, instruction_generation_metrics, scoring_metrics)
        logging.info("Evaluation Report:")
        logging.info(report)

        # Create visualizations
        create_heatmap(document_retrieval_metrics, "Document Retrieval Metrics")
        create_confusion_matrix(scoring_metrics, "Scoring Confusion Matrix")

    except FileNotFoundError as e:
        logging.error(f"File not found error: {str(e)}")
        logging.error("Possible solution: Ensure that the paths to the trained models and test data are correct.")

    except ValueError as e:
        logging.error(f"Value error: {str(e)}")
        logging.error("Possible solution: Check the input data and model architectures for compatibility.")

    except RuntimeError as e:
        logging.error(f"Runtime error: {str(e)}")
        logging.error("Possible solution: Adjust the batch size or model architecture to fit the available memory.")

if __name__ == '__main__':
    main()