from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "gpt2"  # You can use other models like 'gpt2-medium', 'gpt2-large', etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Function to generate text
def generate_text(prompt, max_length=50):
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"], 
            attention_mask=inputs.get("attention_mask"),  # Pass attention mask
            max_length=max_length,  # Maximum length of the generated text
            num_return_sequences=1,  # Number of sequences to return
            no_repeat_ngram_size=2,  # Prevent repetition of n-grams
            pad_token_id=tokenizer.pad_token_id,  # Pad token ID
            do_sample=True,  # Use sampling; set to False for greedy decoding
            top_k=50,  # Top-k sampling
            top_p=0.95,  # Top-p (nucleus) sampling
            temperature=0.7  # Control the randomness of predictions
        )

    # Decode and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the function
if __name__ == "__main__":
    prompt = "Once upon a time"
    generated_text = generate_text(prompt)
    print("Generated Text:\n", generated_text)
