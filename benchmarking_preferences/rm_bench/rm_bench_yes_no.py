import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from datasets import Dataset, load_dataset
from tqdm import tqdm
import re
import gc
import json
import numpy as np

def setup_model(model_id, quantized):
    if quantized:
        print("Loading quantized model...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True,
            offload_folder="offload",
            offload_state_dict=True,
        )
        torch_dtype = torch.bfloat16
        device_map = "auto" if torch.cuda.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def generate_response_entropy(model, tokenizer, prompt, response, debug=False):
    """Calculate entropy only for response tokens, excluding the prompt"""
    try:
        # Ensure text is not empty
        if not response or not response.strip():
            if debug:
                print(f"Warning: Empty response input")
            return float('nan')
        
        if not prompt or not prompt.strip():
            if debug:
                print(f"Warning: Empty prompt input")
            return float('nan')
        
        # Tokenize prompt and full text separately
        prompt_inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        full_text = prompt + response
        full_inputs = tokenizer(full_text, return_tensors="pt", max_length=1024, truncation=True)
        
        prompt_length = prompt_inputs.input_ids.shape[1]
        full_length = full_inputs.input_ids.shape[1]
        
        # Check if we have response tokens
        if full_length <= prompt_length:
            if debug:
                print(f"Warning: No response tokens found (prompt: {prompt_length}, full: {full_length})")
            return float('nan')
        
        response_length = full_length - prompt_length
        
        if debug:
            print(f"Prompt tokens: {prompt_length}")
            print(f"Response tokens: {response_length}")
            print(f"Total tokens: {full_length}")
        
        full_inputs = full_inputs.to(model.device)
        
        with torch.no_grad():
            outputs = model(**full_inputs)
            logits = outputs.logits[0]  # Remove batch dimension
            
            if debug:
                print(f"Logits shape: {logits.shape}")
            
            # Calculate entropy only for response positions
            # Start from prompt_length-1 (to predict first response token) to full_length-1
            entropies = []
            start_pos = max(0, prompt_length - 1)  # Position that predicts first response token
            end_pos = min(logits.shape[0] - 1, full_length - 1)  # Don't go beyond available logits
            
            if debug:
                print(f"Calculating entropy from position {start_pos} to {end_pos}")
            
            for pos in range(start_pos, end_pos):
                # Apply softmax with numerical stability
                logits_pos = logits[pos].float()  # Ensure float32 for stability
                
                if debug and pos < start_pos + 3:
                    print(f"Position {pos} - Logits range: [{torch.min(logits_pos).item():.3f}, {torch.max(logits_pos).item():.3f}]")
                
                # Check for invalid logits
                if torch.isnan(logits_pos).any() or torch.isinf(logits_pos).any():
                    if debug:
                        print(f"Warning: Invalid logits at position {pos}")
                    continue
                
                # Use PyTorch's built-in log_softmax for numerical stability
                log_probs = torch.log_softmax(logits_pos, dim=-1)
                probs = torch.softmax(logits_pos, dim=-1)
                
                if debug and pos < start_pos + 3:
                    print(f"Position {pos} - Probs range: [{torch.min(probs).item():.6f}, {torch.max(probs).item():.6f}]")
                
                # Check for invalid probabilities
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    if debug:
                        print(f"Warning: Invalid probabilities at position {pos}")
                    continue
                
                if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                    if debug:
                        print(f"Warning: Invalid log probabilities at position {pos}")
                    continue
                
                # Calculate entropy: -sum(p * log(p))
                entropy = -torch.sum(probs * log_probs).item()
                
                if debug and pos < start_pos + 3:
                    print(f"Position {pos} - Entropy: {entropy:.6f}")
                
                # Check if entropy is valid
                if not np.isfinite(entropy) or entropy < 0:
                    if debug:
                        print(f"Warning: Invalid entropy at position {pos}: {entropy}")
                    continue
                
                entropies.append(entropy)
            
            if not entropies:
                if debug:
                    print("Warning: No valid response entropies calculated")
                return float('nan')
            
            # Return mean entropy across response sequence
            mean_entropy = sum(entropies) / len(entropies)
            
            if debug:
                print(f"Valid response entropy positions: {len(entropies)}/{end_pos - start_pos}")
                print(f"Response entropies (first 5): {[f'{e:.3f}' for e in entropies[:5]]}")
                print(f"Mean response entropy: {mean_entropy:.6f}")
            
            return mean_entropy
            
    except Exception as e:
        if debug:
            print(f"Error in response entropy calculation: {str(e)}")
            import traceback
            traceback.print_exc()
        return float('nan')

def evaluate_rewards(ds, model, tokenizer, dataset_name, debug_first_few=5):
    levels = [1, 2, 3]
    results = {f'level_{level}': {'correct': 0, 'total': 0} for level in levels}
    processed_data = []
    
    # Track entropy values for initial statistics
    entropy_samples = {f'chosen_{level}': [] for level in levels}
    entropy_samples.update({f'rejected_{level}': [] for level in levels})

    print(f"\n{'='*80}")
    print(f"STARTING RESPONSE ENTROPY CALCULATION FOR: {dataset_name}")
    print("="*80)

    for idx, item in enumerate(tqdm(ds)):
        prompt = item['prompt']
        debug = idx < debug_first_few  # Debug first few entries
        
        if debug:
            print(f"\n=== DEBUGGING ENTRY {idx + 1} ===")
            print(f"Prompt preview: {prompt[:100]}...")
        
        for level in levels:
            chosen_key = f'chosen_{level}'
            rejected_key = f'rejected_{level}'
            
            chosen_response = item[chosen_key]
            rejected_response = item[rejected_key]
            
            if debug:
                print(f"\nLevel {level}:")
                print(f"  Chosen response length: {len(chosen_response)} chars")
                print(f"  Rejected response length: {len(rejected_response)} chars")
                print(f"  Chosen preview: {chosen_response[:150]}...")
                print(f"  Rejected preview: {rejected_response[:150]}...")
            
            # Calculate entropy for both responses (only response tokens, not prompt)
            chosen_entropy = generate_response_entropy(model, tokenizer, prompt, chosen_response, debug=debug and level==1)
            rejected_entropy = generate_response_entropy(model, tokenizer, prompt, rejected_response, debug=debug and level==1)
            
            if debug:
                print(f"  ✓ Chosen response entropy: {chosen_entropy:.4f}" if not np.isnan(chosen_entropy) else f"  ✗ Chosen response entropy: {chosen_entropy}")
                print(f"  ✓ Rejected response entropy: {rejected_entropy:.4f}" if not np.isnan(rejected_entropy) else f"  ✗ Rejected response entropy: {rejected_entropy}")
                
                if not (np.isnan(chosen_entropy) or np.isnan(rejected_entropy)):
                    winner = "CHOSEN" if chosen_entropy < rejected_entropy else "REJECTED"
                    diff = abs(chosen_entropy - rejected_entropy)
                    print(f"  🏆 Winner: {winner} (difference: {diff:.4f})")
            
            # Store entropy values
            item[f'chosen_{level}_entropy'] = chosen_entropy
            item[f'rejected_{level}_entropy'] = rejected_entropy
            
            # Collect samples for statistics (first 20 valid samples)
            if len(entropy_samples[f'chosen_{level}']) < 20 and not np.isnan(chosen_entropy):
                entropy_samples[f'chosen_{level}'].append(chosen_entropy)
            if len(entropy_samples[f'rejected_{level}']) < 20 and not np.isnan(rejected_entropy):
                entropy_samples[f'rejected_{level}'].append(rejected_entropy)
            
            results[f'level_{level}']['total'] += 1
            
            # Lower entropy is better (more confident/certain response)
            # Only count if both entropies are valid
            if not (np.isnan(chosen_entropy) or np.isnan(rejected_entropy)):
                if chosen_entropy < rejected_entropy:
                    results[f'level_{level}']['correct'] += 1
            elif debug:
                print(f"  ⚠️  Skipping comparison due to nan values")

        processed_data.append(item)
        
        # Print entropy statistics after first 20 samples
        if idx == 19:  # After 20 entries
            print(f"\n{'='*60}")
            print("RESPONSE ENTROPY STATISTICS (First 20 valid samples):")
            print("="*60)
            
            for level in levels:
                chosen_samples = entropy_samples[f'chosen_{level}']
                rejected_samples = entropy_samples[f'rejected_{level}']
                
                print(f"\nLevel {level}:")
                if chosen_samples:
                    print(f"  Chosen   - Count: {len(chosen_samples):2d}, Mean: {np.mean(chosen_samples):.4f}, Std: {np.std(chosen_samples):.4f}")
                    print(f"             Range: [{min(chosen_samples):.4f}, {max(chosen_samples):.4f}]")
                else:
                    print(f"  Chosen   - No valid samples yet")
                
                if rejected_samples:
                    print(f"  Rejected - Count: {len(rejected_samples):2d}, Mean: {np.mean(rejected_samples):.4f}, Std: {np.std(rejected_samples):.4f}")
                    print(f"             Range: [{min(rejected_samples):.4f}, {max(rejected_samples):.4f}]")
                else:
                    print(f"  Rejected - No valid samples yet")
                
                if chosen_samples and rejected_samples:
                    mean_diff = np.mean(rejected_samples) - np.mean(chosen_samples)
                    print(f"  Difference (Rejected - Chosen): {mean_diff:.4f}")
                    if mean_diff > 0:
                        print(f"  📊 Chosen responses have LOWER entropy (more confident) ✓")
                    else:
                        print(f"  📊 Rejected responses have LOWER entropy (more confident) ⚠️")
            
            print("="*60)

    accuracies = {
        level: (results[level]['correct'] / results[level]['total']) * 100 
        if results[level]['total'] > 0 else 0 
        for level in results
    }

    return accuracies, processed_data

def save_entropy_data(processed_data, dataset_name, model_name):
    """Save entropy data to JSON file for each dataset"""
    # Extract dataset name for filename
    name_match = re.search(r'/([^/]+)$', dataset_name)
    if name_match:
        name = name_match.group(1)
    else:
        name = dataset_name.replace('/', '_')
    
    filename = f"entropy_data_{name}_{model_name.split('/')[-1]}.json"
    
    # Convert nan values to null for JSON serialization
    def convert_nan(obj):
        if isinstance(obj, dict):
            return {k: convert_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_nan(v) for v in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        else:
            return obj
    
    processed_data_clean = convert_nan(processed_data)
    
    with open(filename, 'w') as f:
        json.dump(processed_data_clean, f, indent=2)
    print(f"Entropy data saved to {filename}")

def save_all_accuracies_to_json(all_accuracies, model_name):
    """Save accuracies to JSON file"""
    filename = f"accuracy-rm-bench-{model_name.split('/')[-1]}-entropy.json"
    with open(filename, 'w') as f:
        json.dump(all_accuracies, f, indent=4)
    print(f"All accuracies saved to {filename}")

def save_combined_entropy_data(all_processed_data, model_name):
    """Save all entropy data combined into one file"""
    filename = f"combined_entropy_data_{model_name.split('/')[-1]}.json"
    
    # Convert nan values to null for JSON serialization
    def convert_nan(obj):
        if isinstance(obj, dict):
            return {k: convert_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_nan(v) for v in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        else:
            return obj
    
    all_processed_data_clean = convert_nan(all_processed_data)
    
    with open(filename, 'w') as f:
        json.dump(all_processed_data_clean, f, indent=2)
    print(f"Combined entropy data saved to {filename}")

def main(args):
    login(args.hf_key)
    model, tokenizer = setup_model(args.model_name, args.quantized)
    
    # Test entropy calculation with a simple example
    print("Testing response entropy calculation...")
    test_prompt = "What is the capital of France?"
    test_response = " The capital of France is Paris."
    test_entropy = generate_response_entropy(model, tokenizer, test_prompt, test_response, debug=True)
    print(f"Test response entropy: {test_entropy}")
    
    datasets = [
        "Ayush-Singh/RM-Bench-chat",
        "Ayush-Singh/RM-Bench-code", 
        "Ayush-Singh/RM-Bench-math",
        "Ayush-Singh/RM-Bench-safety-response",
        "Ayush-Singh/RM-Bench-safety-refuse",
    ]
    
    all_accuracies = {}
    all_processed_data = []
    
    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)['train']
        accuracies, processed_data = evaluate_rewards(dataset, model, tokenizer, dataset_name)
        
        all_accuracies[dataset_name] = accuracies
        all_processed_data.extend(processed_data)
        
        for level, acc in accuracies.items():
            print(f"Accuracy for {dataset_name} - {level}: {acc:.2f}%")
        
        # Save entropy data for this dataset
        save_entropy_data(processed_data, dataset_name, args.model_name)
        
        # Extract dataset name for local saving
        name_match = re.search(r'/([^/]+)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer rewards using entropy and push results to Hugging Face Hub")
    parser.add_argument("--hf_key", type=str, required=True, help="Hugging Face API key")
    parser.add_argument("--hf_user", type=str, required=True, help="Hugging Face user name to push datasets")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model on Hugging Face")
    parser.add_argument("--quantized", action="store_true", help="Use quantized model for inference")
    args = parser.parse_args()

    main(args), dataset_name)
        if name_match:
            name = name_match.group(1)
        else:
            name = dataset_name.replace('/', '_')
        
        # Create processed dataset
        processed_dataset = Dataset.from_list(processed_data)
        
        # Save locally instead of pushing to hub (for testing)
        local_path = f"{name}-{args.model_name.split('/')[-1]}-response-entropy"
        processed_dataset.save_to_disk(local_path)
        print(f"Dataset saved locally to: {local_path}")
        
        # Uncomment below when HF token permissions are fixed
        # processed_dataset.push_to_hub(f"{args.hf_user}/{name}-{args.model_name.split('/')[-1]}-response-entropy")

    save_all_accuracies_to_json(all_accuracies, args.model_name)
    save_combined_entropy_data(all_processed_data, args.model_name)
    
    del model
    gc.collect()
    
    return all_processed_data, datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer rewards using entropy and push results to Hugging Face Hub")
    parser.add_argument("--hf_key", type=str, required=True, help="Hugging Face API key")
    parser.add_argument("--hf_user", type=str, required=True, help="Hugging Face user name to push datasets")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model on Hugging Face")
    parser.add_argument("--quantized", action="store_true", help="Use quantized model for inference")
    args = parser.parse_args()

    main(args)
