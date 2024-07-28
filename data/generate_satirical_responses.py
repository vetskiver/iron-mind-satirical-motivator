import json 
from transformers import pipeline, set_seed

# Set the seed for reproducibility
set_seed(42)

# Load JSON data
with open('/Users/alirezaghasemi/Desktop/iron-mind-satirical-motivator/daily_activity_logs.json', 'r') as file:
    data_entries = json.load(file)

# Base prompt for generating responses
base_prompt = """
The user has provided their daily activity log. Generate a satirical response that is motivational, funny, and humorous, delivered in a brash and ranting tone similar to David Goggins and Andrew Tate. The response should address the user's Physical Health, Mental Health, Work/Study, and Interesting Events in that order. Be sure to cover each aspect with specific comments based on the user's input. The response should be structured as follows:

Physical Health
Steps: Comment on the number of steps taken. If low, make a humorous jab at their lack of movement. If high, give exaggerated praise.
Calories Burned: React to the calories burned. If low, mock their effort. If high, give an over-the-top compliment.
Workouts: Discuss the workout intensity. If none, mock their laziness. If intense, shower them with brash praise.
Diet: Critique their diet choices. If unhealthy, use humor to point out poor decisions. If healthy, praise with a humorous twist.
Sleep: Comment on their sleep duration. If short, make a humorous remark about their zombie-like state. If sufficient, praise their rest with a funny spin.
Water Intake: React to their water consumption. If low, mock their hydration efforts. If high, give exaggerated approval.

Mental Health
Mood: Comment on their mood. If low, use humor to motivate them to improve. If high, give a brash compliment.
Social Interactions: React to their level of social interactions. If none, mock their isolation humorously. If many, praise their social efforts with a funny twist.
Screen Time: Discuss their screen time. If high, mock their screen addiction. If low, praise their balance humorously.
Relaxation: Comment on their relaxation time. If none, humorously point out their stress. If some, praise their relaxation efforts.

Work/Study
Productivity: React to their productivity level. If low, mock their laziness. If high, give exaggerated praise.

Interesting Events
Events: Comment on the interesting events of their day. Use humor and satire to make the response engaging and funny.

Ensure the response is cohesive, flows well, and maintains the satirical and brash tone throughout.
"""

# Initialize the text generation pipeline
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

def generate_response(entry):
    # Create a specific prompt for the entry
    prompt = f"{base_prompt}\nUser Input:\n"
    for key, value in entry.items():
        prompt += f"{key}: {value}\n"
    prompt += "\nResponse:\n"

    # Generate response using the model
    response = generator(prompt, max_length=400, num_return_sequences=1)

    return response[0]['generated_text'].strip()

# Generate a response for the first entry
first_entry = data_entries[0]
response = generate_response(first_entry)

print(response)

# Save the response to a file
with open('generated_response.txt', 'w') as file:
    file.write(response + "\n" + "-"*50 + "\n")