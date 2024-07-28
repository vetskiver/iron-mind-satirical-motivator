import json
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Load the JSON data
with open('/path/to/your/daily_activity_logs.json', 'r') as file:
    data_entries = json.load(file)

# Convert the data into a list of strings suitable for GPT-2 fine-tuning
data_strings = []

base_prompt = """
Generate a satirical, motivational, and humorous response in the style of David Goggins and Andrew Tate, 
based on the user's daily activity log. Address Physical Health, Mental Health, Work/Study, and Interesting 
Events with specific comments.

Physical Health
- Steps: Praise or mock based on step count.
- Calories Burned: Compliment or mock based on calories burned.
- Workouts: Praise intensity or mock laziness.
- Diet: Critique humorously.
- Sleep: Comment on sleep duration with humor.
- Water Intake: Mock or praise based on intake.

Mental Health
- Mood: Motivate or compliment.
- Social Interactions: Mock isolation or praise socializing.
- Screen Time: Mock high usage or praise balance.
- Relaxation: Point out stress or praise relaxation.

Work/Study
- Productivity: Mock laziness or praise productivity.

Interesting Events
- Comment humorously on daily events.

Keep the tone satirical, brash, and cohesive.
"""

# Combine base_prompt with each entry to create training data
for entry in data_entries:
    entry_str = f"{base_prompt}\nUser Input:\n"
    for key, value in entry.items():
        entry_str += f"{key}: {value}\n"
    entry_str += "\nResponse:\n"
    data_strings.append(entry_str)

# Create a Hugging Face dataset
dataset = Dataset.from_dict({"text": data_strings})

# Load the pre-trained GPT-2 model and tokenizer
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,  # Use mixed precision training for faster training if available
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=10,
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,  # For simplicity, using the same data for evaluation
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("/Users/alirezaghasemi/Desktop/iron-mind-satirical-motivator/model/fine-tuned-model")
tokenizer.save_pretrained("/Users/alirezaghasemi/Desktop/iron-mind-satirical-motivator/model/fine-tuned-model")