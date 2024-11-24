import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

# Environment
SLURM = True
DEBUG = True 

# Define the debug print function
def print_debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# Define a custom print function
def print_flush(*args, **kwargs):
    __builtins__.print(*args, **kwargs)
    sys.stdout.flush()

try:
    from plot_results import plot_tokens_with_colored_background, plot_sentences_with_colored_background
except ImportError:
    print("Warning: Unable to import 'plot_results'. Placeholder functions will be used.")
    # Define placeholder functions if the module is not available
    def plot_tokens_with_colored_background(*args, **kwargs):
        print("plot_tokens_with_colored_background is unavailable.")

    def plot_sentences_with_colored_background(*args, **kwargs):
        print("plot_sentences_with_colored_background is unavailable.")

try:
    from model_assesment_data_set import gather_all_prompts
except ImportError:
    print("Warning: Unable to import 'nature_facts_data_set'. Placeholder functions will be used.")
    # Define placeholder function if the module is not available
    def gather_all_prompts(*args, **kwargs):
        print("gather_all_prompts is unavailable.")
        return []

try:
    from data_handling import split_paragraph_to_sentences_with_offsets, map_tokens_to_sentence_indices
except ImportError:
    print("Warning: Unable to import 'data_handling'. Placeholder functions will be used.")
    # Define placeholder functions if the module is not available
    def split_paragraph_to_sentences_with_offsets(*args, **kwargs):
        print("split_paragraph_to_sentences_with_offsets is unavailable.")
        return [], []

    def map_tokens_to_sentence_indices(*args, **kwargs):
        print("map_tokens_to_sentence_indices is unavailable.")
        return [], []

try:
    from needle_in_a_haystack_data_set import get_all_needle_texts, get_all_needles
except ImportError:
    print("Warning: Unable to import 'needle_in_a_haystack_data_set'. Placeholder functions will be used.")
    # Define placeholder functions if the module is not available
    def get_all_needle_texts(*args, **kwargs):
        print("get_all_needle_texts is unavailable.")
        return []

    def get_all_needles(*args, **kwargs):
        print("get_all_needles is unavailable.")
        return []

try:
    from slurm_environment import set_os_environ, get_pathes
except ImportError:
    print("Warning: Unable to import 'slurm_environment'. Placeholder functions will be used.")
    # Define placeholder functions if the module is not available
    def set_os_environ(*args, **kwargs):
        print("set_os_environ is unavailable.")

    def get_pathes(*args, **kwargs):
        print("get_pathes is unavailable.")

use_cude = SLURM
current_path_token = None
current_path_sentence = None
if SLURM:
    print("#####################")
    print("Setting slurm profile")
    print("#####################")
    try:
        set_os_environ()
        current_path_token, current_path_sentence = get_pathes()
        # Override the built-in print function
        print = print_flush
    except Exception as e:
        print(f"Error during SLURM environment setup: {e}")
else:
    print("#############s########")
    print("Setting local profile")
    print("#####################")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

eps = 1e-8

# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace with your actual Hugging Face token or handle it securely
token = None
if token: 
    login(token=token)
else:
    print("Error: Please provide a valid Hugging Face login token.")

def append_next_token(tokens, next_token):
    if not isinstance(next_token, torch.Tensor):
        next_token = torch.tensor([next_token], device=tokens.device)
    tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=-1)
    return tokens

def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam

def apply_self_attention_rules(R_ss, cam_ss):
    # Ensure R_ss and cam_ss have the same dtype
    R_ss = R_ss.to(dtype=cam_ss.dtype)
    return torch.matmul(cam_ss, R_ss)

def generate_relevance(model, input_tensor, index=None):
    input_tensor = input_tensor.to(device)
    if use_cude:
        with torch.amp.autocast(device_type='cuda'):
            output = model(input_tensor)
            next_token_logits = output.logits[:, -1, :]
    else:
        output = model(input_tensor)
        next_token_logits = output.logits[:, -1, :]

    if index is None:
        index = torch.argmax(next_token_logits).item()

    one_hot = torch.zeros_like(next_token_logits).to(device)
    one_hot[0, index] = 1
    one_hot = torch.sum(one_hot * next_token_logits)

    # Compute gradients efficiently without retaining the computation graph
    grads = torch.autograd.grad(one_hot, output.attentions, retain_graph=False, create_graph=False)

    num_tokens = output.attentions[0].shape[-1]

    # Initialize R with the same dtype as attention tensors
    R = torch.eye(num_tokens, num_tokens, dtype=output.attentions[0].dtype, device=device)

    for attn, grad in zip(output.attentions, grads):
        # Apply average heads function to combine gradients and attention maps
        cam = avg_heads(attn, grad)
        # Update the relevance matrix using self-attention rules
        R += apply_self_attention_rules(R, cam)

    # Return the relevance scores, excluding the first token (often a padding token)
    return R, index

class LlmExplainablilityModel:    
    def __init__(self, model_name="google/gemma-2-2b-it"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            output_attentions=True, 
            attn_implementation="eager"
        )
        self.model.eval()  
        self.model.half()
        self.model.to(device)  
        self.prompt_number = 1

    def generate_text_from_tokens(self, tokens):
        text_from_tokens = []
        for token in tokens:
            token_text = self.tokenizer.decode([token.item()], skip_special_tokens=False)
            text_from_tokens.append(token_text)
        return text_from_tokens

    def generate_text(self, input_ids, max_generation_len, print_generated_ids=False, print_generated_text=False):
        input_ids = input_ids.to(device)
        if use_cude:
            with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                generated_ids = self.model.generate(input_ids, max_new_tokens=max_generation_len)
        else:
            with torch.no_grad():
                generated_ids = self.model.generate(input_ids, max_new_tokens=max_generation_len)

        if print_generated_ids:
            print("Printing generated_ids:")
            print(generated_ids)
            print("End printing generated_ids.")

        # Decode the generated token IDs to text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        if print_generated_text:
            print("The generated text is:", generated_text)

        return generated_ids, generated_text

    def print_sorted_indices(self, tensor_values):
        # Sort the tensor in descending order and get the sorted indices
        sorted_indices = torch.argsort(tensor_values, descending=True)
        print(f"Sorted indices (descending by value): {sorted_indices}")

    def print_plot_data(self, tokens, token_scores, max_tokens_per_line=10):
        num_lines = (len(tokens) + max_tokens_per_line - 1) // max_tokens_per_line

        print("Token Plot Data:")
        for line in range(num_lines):
            start_idx = line * max_tokens_per_line
            end_idx = min((line + 1) * max_tokens_per_line, len(tokens))
            tokens_line = tokens[start_idx:end_idx]
            scores_line = token_scores[start_idx:end_idx]

            # Format each line for better readability
            line_str = " | ".join(f"{token}: {score:.2f}" for token, score in zip(tokens_line, scores_line))
            print(f"Line {line + 1}: {line_str}")

    def compute_average_by_sentence(self, relevance_matrix, sentence_indices):
        sentence_indices = torch.tensor(sentence_indices).to(device)

        # Get the unique sentence indices
        unique_indices = torch.unique(sentence_indices)

        # Compute the average value for each unique sentence index
        for idx in unique_indices:
            mask = sentence_indices == idx
            avg_value = relevance_matrix[mask].mean()
            relevance_matrix[mask] = avg_value

        return relevance_matrix

    def generate_relevance_for_tokens(self, input_text, tokens, max_generation_len):
        num_tokens = tokens.size()[-1] - 1
        relevance_matrix = torch.zeros(num_tokens, dtype=torch.float32, device=device)

        first = True
        print_debug("----------")
        print_debug("Printing loop process: ")
        for i in range(max_generation_len):
            if i % 5 == 0:
                if first:
                    print_debug(f'The loop index is: {i}', end=", ")
                    first = False
                else:
                    print_debug(f'{i}', end=", ")
            R, next_token = generate_relevance(model=self.model, input_tensor=tokens)

            if i > 0:
                last_line = R[-1, 1:num_tokens + 1].detach().cpu().float().numpy()
                last_line = last_line / (sum(last_line) + eps)
                relevance_matrix += torch.tensor(last_line, dtype=torch.float32).to(device)

            text_from_tokens = self.generate_text_from_tokens(tokens[0])
            token_text = self.tokenizer.decode([next_token], skip_special_tokens=False)
            heat_map_prob = R[-1, 1:]
            heat_map_prob[-1] -= 1
            if use_cude:
                torch.cuda.empty_cache()

            if next_token == 1 or next_token == 107:
                print_debug(f'Broke here index: {i}', end="")
                break

            tokens = append_next_token(tokens, next_token)

            if use_cude:
                # Free up memory
                del R, next_token, heat_map_prob
                torch.cuda.empty_cache()

        relevance_matrix_normalized = relevance_matrix / (sum(relevance_matrix) + eps)

        # Plot by tokens
        plot_tokens_with_colored_background(text_from_tokens[1:], relevance_matrix_normalized, self.prompt_number, current_path_token)

        # Plot by sentences
        inputs, sentence_indices = map_tokens_to_sentence_indices(self.tokenizer, input_text)
        print_debug("inputs:")
        print_debug(inputs)
        print_debug("sentence_indices:")
        print_debug(sentence_indices)
        sentence_relevance = self.compute_average_by_sentence(relevance_matrix_normalized, sentence_indices[1:])
        sentence_relevance = sentence_relevance / (sum(sentence_relevance) + eps)

        plot_sentences_with_colored_background(text_from_tokens[1:], sentence_relevance, sentence_indices[1:], self.prompt_number, current_path_sentence)
        self.prompt_number += 1

        print_debug()
        print_debug("----------")
        generated_ids, generated_text = self.generate_text(tokens.to(device), max_generation_len, False, True)
        return tokens, generated_ids, generated_text

    def get_attention_and_gradients(self, input_text, max_generation_len=50):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)

        tokens, generated_ids, generated_text = self.generate_relevance_for_tokens(input_text, input_ids, max_generation_len)
        return generated_text

def run_explainability_model(explainability_model, prompt):
    explainability_model.get_attention_and_gradients(prompt)

def init_model():
    print("Initializing model")

    explainability_model = LlmExplainablilityModel(model_name="google/gemma-2-2b-it")

    print("Finished initializing model.")

    return explainability_model

def start_main_flow():
    explainability_model = init_model()

    inputs = gather_all_prompts()

    for idx, prompt in enumerate(inputs):
        print("----------------------------------------------------------------------")
        print(f"Running explainability model for prompt: {prompt}")
        generated_text = run_explainability_model(explainability_model, prompt)

if __name__ == "__main__":
    start_main_flow()
