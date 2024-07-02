from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import pandas as pd

# Sample dataset
# conversations = [
#     {"input_text": "User: My client wants to open a new account", "target_text": "Bot: Is your client interested in a Controlled Disbursement Account (CDA) or Money Market Deposit Accounts (MMDA)"},
#     {"input_text": "User: Can you explain the differences between the services?", "target_text": "Bot: Controlled Disbursement Account (CDA) provides accurate disbursement totals early each business day to gain control over idle balances, eliminate overdrafts and automate funding transfers. Money Market Deposit Accounts (MMDA) clients can use this interest-bearing solution to set aside reserve cash to fund periodic payments such as capital improvements, lease payments, etc. or simply to provide a liquidity cushion to accommodate a sudden need for cash."},
#     {"input_text": "User: What service could help with managing internal and external accounts?", "target_text": "Bot: Multi-Bank Reporting, trovata, and Multi-Bank Sweep"},
#     {"input_text": "User: What service could help with managing JPMC and Wells Fargo Accounts?", "target_text": "Bot: Multi-Bank Reporting, trovata, and Multi-Bank Sweep"},
#     {"input_text": "User: what is Multi-Bank Sweep?", "target_text": "Bot: Multi-Bank Sweep automatically moves balances from an account in one bank to an account in another bank. This product automates the movement of cash between client-owned accounts at JPM and third-party banks via SWIFT messaging."},
#     {"input_text": "User: what is Multi-Bank Reporting?", "target_text": "Bot: Multi-Bank Reporting provides full visibility of all J.P. Morgan accounts globally as well as accounts with third-party banks who send reporting data to JPM."},
#     {"input_text": "User: what is trovata?", "target_text": "Bot: trovata is JPM FinTech partnership. Multi-bank cash management platform that helps clients with cash forecasting, cash flow analysis, and cash reporting. Available as a paid add-on to J.P. Morgan Access."}
# ]

conversations = [
    {"input_text": "My client wants to open a new account", "target_text": "Is your client interested in a Controlled Disbursement Account (CDA) or Money Market Deposit Accounts (MMDA)"},
    {"input_text": "Can you explain the differences between the services?", "target_text": "Controlled Disbursement Account (CDA) provides accurate disbursement totals early each business day to gain control over idle balances, eliminate overdrafts and automate funding transfers. Money Market Deposit Accounts (MMDA) clients can use this interest-bearing solution to set aside reserve cash to fund periodic payments such as capital improvements, lease payments, etc. or simply to provide a liquidity cushion to accommodate a sudden need for cash."},
    {"input_text": "What service could help with managing internal and external accounts?", "target_text": "Multi-Bank Reporting, trovata, and Multi-Bank Sweep"},
    {"input_text": "What service could help with managing JPMC and Wells Fargo Accounts?", "target_text": "Multi-Bank Reporting, trovata, and Multi-Bank Sweep"},
    {"input_text": "What is Multi-Bank Sweep?", "target_text": "Multi-Bank Sweep automatically moves balances from an account in one bank to an account in another bank. This product automates the movement of cash between client-owned accounts at JPM and third-party banks via SWIFT messaging."},
    {"input_text": "what is Multi-Bank Reporting?", "target_text": "Multi-Bank Reporting provides full visibility of all J.P. Morgan accounts globally as well as accounts with third-party banks who send reporting data to JPM."},
    {"input_text": "what is trovata?", "target_text": "Trovata is JPM FinTech partnership. Multi-bank cash management platform that helps clients with cash forecasting, cash flow analysis, and cash reporting. Available as a paid add-on to J.P. Morgan Access."}
]

# Convert to DataFrame
df = pd.DataFrame(conversations)
df.to_csv('/conversations.csv', index=False)

# Load the dataset
dataset = Dataset.from_pandas(df)

# Split the dataset into training and validation sets (e.g., 90% train, 10% validation)
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# Load the tokenizer and model
model_name = "EleutherAI/gpt-neo-1.3B"  # Use GPT-Neo model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name,ignore_mismatched_sizes=True)

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples["input_text"], truncation=True, padding="max_length", max_length=128)
    targets = tokenizer(examples["target_text"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Define data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('./fine-tuned-model')
tokenizer.save_pretrained('./fine-tuned-tokenizer')

# Load the fine-tuned model and tokenizer
fine_tuned_model = GPTNeoForCausalLM.from_pretrained('./fine-tuned-model',ignore_mismatched_sizes=True)
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained('./fine-tuned-tokenizer')

# Ensure model is in evaluation mode
fine_tuned_model.eval()

# Sample inputs from your dataset to validate
sample_inputs = [
    "User: My client wants to open a new account",
    "User: Can you explain the differences between the services?",
    "User: What service could help with managing internal and external accounts?",
    "User: What service could help with managing JPMC and Wells Fargo Accounts?",
    "User: what is Multi-Bank Sweep?",
    "User: what is Multi-Bank Reporting?",
    "User: what is trovata?"
]

# Function to generate response
def generate_response(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=128, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate responses for the sample inputs
for input_text in sample_inputs:
    response = generate_response(fine_tuned_model, fine_tuned_tokenizer, input_text)
    print(f"Input: {input_text}")
    print(f"Response: {response}")
    print()
