# instruction_tuned_qa.py

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset

def load_data(dataset_name, subset=None):
    """
    Load and preprocess the dataset for instruction-tuned QA.

    Args:
        dataset_name (str): The name of the dataset to load (e.g., "squad", "trivia_qa").
        subset (str, optional): The subset of the dataset to use (e.g., "train", "validation"). Defaults to None.

    Returns:
        list: A list of dictionaries containing the context, question, and answer for each example.
    """
    try:
        # Load the dataset using the Hugging Face datasets library
        dataset = load_dataset(dataset_name, split=subset)

        # Preprocess the dataset
        processed_data = []
        for example in dataset:
            context = example["context"]
            question = example["question"]
            answer = example["answer"]

            # Perform any necessary preprocessing steps
            # Example: Truncate the context to a maximum length
            max_context_length = 512
            context = context[:max_context_length]

            processed_example = {
                "context": context,
                "question": question,
                "answer": answer
            }
            processed_data.append(processed_example)

        return processed_data

    except ValueError as e:
        print(f"Error: Invalid dataset name or subset: {str(e)}")
        return None

    except KeyError as e:
        print(f"Error: Required keys not found in the dataset: {str(e)}")
        return None

    except Exception as e:
        print(f"Error: An unexpected error occurred during data loading: {str(e)}")
        return None

def generate_instruction_tuned_qa(context, model_name="t5-base", num_beams=4, num_return_sequences=1):
    """
    Generate question-answer pairs based on the given context using an instruction-tuned T5 model.

    Args:
        context (str): The input context for generating question-answer pairs.
        model_name (str, optional): The name of the pre-trained T5 model to use. Defaults to "t5-base".
        num_beams (int, optional): The number of beams to use for beam search decoding. Defaults to 4.
        num_return_sequences (int, optional): The number of sequences to generate. Defaults to 1.

    Returns:
        list: A list of generated question-answer pairs.
    """
    try:
        # Load the pre-trained T5 model and tokenizer
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Define the instruction template for generating question-answer pairs
        instruction_template = "Generate a question-answer pair based on the following context:\n{context}\n\nQuestion: "

        # Tokenize the input context and instruction
        input_text = instruction_template.format(context=context)
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Generate question-answer pairs using beam search decoding
        output_ids = model.generate(
            input_ids,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            max_length=100,
            early_stopping=True
        )

        # Decode the generated output
        generated_qa_pairs = []
        for output_id in output_ids:
            output_text = tokenizer.decode(output_id, skip_special_tokens=True)
            qa_pair = output_text.split("Question: ")[-1].split("Answer: ")
            question = qa_pair[0].strip()
            answer = qa_pair[1].strip() if len(qa_pair) > 1 else ""
            generated_qa_pairs.append({"question": question, "answer": answer})

        return generated_qa_pairs

    except ValueError as e:
        print(f"Error: Invalid model name or configuration: {str(e)}")
        return None

    except RuntimeError as e:
        print(f"Error: An error occurred during model execution: {str(e)}")
        return None

    except Exception as e:
        print(f"Error: An unexpected error occurred during QA generation: {str(e)}")
        return None

def evaluate_generated_qa(generated_qa_pairs, reference_qa_pairs):
    """
    Evaluate the quality of the generated question-answer pairs using metrics like BLEU and ROUGE.

    Args:
        generated_qa_pairs (list): A list of generated question-answer pairs.
        reference_qa_pairs (list): A list of reference question-answer pairs.

    Returns:
        dict: A dictionary containing the evaluation metrics (e.g., BLEU scores, ROUGE scores).
    """
    # Implement evaluation metrics like BLEU and ROUGE
    # You can use libraries like nltk or rouge to calculate the scores

    # Example evaluation using BLEU score
    from nltk.translate.bleu_score import corpus_bleu

    reference_questions = [[qa["question"].split()] for qa in reference_qa_pairs]
    generated_questions = [qa["question"].split() for qa in generated_qa_pairs]

    bleu_scores = corpus_bleu(reference_questions, generated_questions)

    # You can calculate other evaluation metrics similarly

    evaluation_metrics = {
        "bleu_scores": bleu_scores,
        # Add other evaluation metrics
    }

    return evaluation_metrics

def main():
    # Load the dataset
    dataset_name = "squad"
    subset = "train"
    data = load_data(dataset_name, subset)

    if data is None:
        print("Error: Failed to load data.")
        return

    # Generate question-answer pairs using the instruction-tuned model
    generated_qa_pairs = []
    for example in data:
        context = example["context"]
        qa_pairs = generate_instruction_tuned_qa(context, model_name="t5-base", num_return_sequences=3)
        generated_qa_pairs.extend(qa_pairs)

    # Evaluate the generated question-answer pairs
    reference_qa_pairs = [{"question": example["question"], "answer": example["answer"]} for example in data]
    evaluation_metrics = evaluate_generated_qa(generated_qa_pairs, reference_qa_pairs)

    # Print the evaluation metrics
    print("Evaluation Metrics:")
    for metric, score in evaluation_metrics.items():
        print(f"{metric}: {score}")

if __name__ == "__main__":
    main()
