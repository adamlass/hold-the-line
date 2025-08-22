import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_best_device

# Set up device
device = get_best_device()

# Load the Llama model and tokenizer (use your model checkpoint)
# MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Set model in training mode and define optimizer
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)

def online_data_generator():
    """
    Simulate an online data stream.
    Replace this generator with your actual data source.
    """
    while True:
        # For example, this prompt could be replaced by real-time user input or sensor data.
        prompt = "The quick brown fox jumps over the lazy dog."
        yield prompt

def calculate_reward(generated_text):
    """
    Manually calculate a reward based on the generated text.
    Replace this logic with your own reward calculation.
    """
    # For demonstration, we simply return a fixed reward.
    # In a realistic scenario, the reward might depend on correctness,
    # adherence to instructions, or any other criteria.
    return torch.tensor(1.0, device=device)

# Initialize online data stream
data_stream = online_data_generator()

# Example training loop over online data
for step in range(10):  # This loop can run indefinitely in an online scenario.
    # Get new online data
    prompt = next(data_stream)
    
    # Tokenize the prompt and move inputs to device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate a response from the model (this can be modified to generate step-by-step)
    output_ids = model.generate(**inputs, max_new_tokens=10)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Calculate a custom reward for the generated output
    reward = calculate_reward(generated_text)
    
    # --- Forward Pass for Loss Calculation ---
    # Compute the standard language modeling loss using teacher forcing
    outputs = model(**inputs, labels=inputs["input_ids"])
    lm_loss = outputs.loss  # This is the cross entropy loss for language modeling
    
    # Here, we manually incorporate the reward into the loss.
    # For example, we might scale the loss by the negative reward:
    # (This is a simplistic demonstration; in practice, you might use a PPO-style loss or another RL algorithm.)
    combined_loss = lm_loss - reward * 0.1  # The scaling factor (0.1) is arbitrary
    
    # Backpropagation and optimizer step
    optimizer.zero_grad()
    combined_loss.backward()
    optimizer.step()
    
    print(f"Step {step}:")
    print(f"  Prompt: {prompt}")
    print(f"  Generated Text: {generated_text}")
    print(f"  Language Modeling Loss: {lm_loss.item():.4f}")
    print(f"  Reward: {reward.item():.4f}")
    print(f"  Combined Loss: {combined_loss.item():.4f}\n")