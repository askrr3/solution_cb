import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict, load_dataset

# Define a custom Trainer subclass
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["input_ids"]  # Assuming your labels are in input_ids
        
        outputs = model(**inputs)
        logits = outputs.logits

        # Calculate the CrossEntropy loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# Function to prepare dataset
def prepare_dataset(tokenizer, train_dataset, test_dataset):
    # Tokenize the datasets
    def tokenize_function(examples):
        return tokenizer(examples["description"], padding="max_length", truncation=True, max_length=512)

    # Convert the datasets to DatasetDict objects
    tokenized_datasets = DatasetDict({
        "train": Dataset.from_dict(train_dataset),
        "test": Dataset.from_dict(test_dataset)
    })

    # Apply tokenization function to the datasets
    tokenized_datasets = tokenized_datasets.map(tokenize_function, batched=True)

    return tokenized_datasets

# Function to fine-tune the model
def fine_tune_model(model, tokenizer, tokenized_datasets):
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    trainer.train()
    model.save_pretrained("./fine-tuned-llama2")
    tokenizer.save_pretrained("./fine-tuned-llama2")

# Function to generate a response
def generate_response(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Main function to coordinate the training and inference process
def main():
    try:
        model_name = "meta-llama/Meta-Llama-Guard-2-8B"
        
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Load your train and test datasets (replace with your actual data loading logic)
        train_dataset = [
            {"service": "RideSharing", "description": "A service to book rides with drivers in your area"},
            {"service": "FoodDelivery", "description": "A service to order food from various restaurants and have it delivered to your door"},
            # Add more examples as needed
        ]

        test_dataset = [
            {"service": "GroceryDelivery", "description": "A service to order groceries online and have them delivered to your home"},
            {"service": "HomeRepair", "description": "A service to find and book professionals for various home repair tasks"},
            # Add more examples as needed
        ]

        print("Preparing dataset...")
        tokenized_datasets = prepare_dataset(tokenizer, train_dataset, test_dataset)

        print("Fine-tuning the model...")
        fine_tune_model(model, tokenizer, tokenized_datasets)

        print("Welcome to the service assistant. How can I help you today?")
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            prompt = f"User: {user_input}\nAssistant:"
            response = generate_response(prompt, model, tokenizer)
            print(f"Assistant: {response}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
