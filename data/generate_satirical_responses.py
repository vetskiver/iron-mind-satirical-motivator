import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# Load the JSON data
with open('/Users/alirezaghasemi/Desktop/iron-mind-satirical-motivator/daily_activity_logs.json', 'r') as file:
    data_entries = json.load(file)

# Prepare the base prompt
base_prompt = """
Generate a satirical, motivational, and humorous response in the style of David Goggins and Andrew Tate, 
based on the user's daily activity log. Address Physical Health, Mental Health, Work/Study, and Interesting 
Events with specific comments.

Physical Health
- Steps: Praise or mock based on step count.
- Calories Burned: Compliment or mock based on calories burned.
- Workouts: Praise intensity or mock laziness.
- Diet: Critique humorously.
- Sleep: Comment on sleep duration with hmor.
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

# Combine base_prompt with each entry to create the training data
data_strings = []
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

# Add padding token to the tokenizer
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    num_train_epochs=3,  # Adjust the number of epochs based on your dataset size and computational resources
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=10_000,  # Adjust this based on your dataset size
    save_total_limit=2,  # Limit the total number of checkpoints
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("/Users/alirezaghasemi/Desktop/iron-mind-satirical-motivator/model/fine-tuned-model")
tokenizer.save_pretrained("/Users/alirezaghasemi/Desktop/iron-mind-satirical-motivator/model/fine-tuned-model")
