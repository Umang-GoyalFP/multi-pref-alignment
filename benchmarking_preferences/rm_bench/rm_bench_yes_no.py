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

def generate_entropy(model, tokenizer, text):
    """Calculate mean entropy across the entire sequence"""
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]  # Remove batch dimension
        
        # Calculate entropy for each position
        entropies = []
        for pos in range(logits.shape[0]):
            probs = torch.softmax(logits[pos], dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
            entropies.append(entropy)
        
        # Return mean entropy across sequence
        return sum(entropies) / len(entropies)

def evaluate_rewards(ds, model, tokenizer, dataset_name):
    levels = [1, 2, 3]
    results = {f'level_{level}': {'correct': 0, 'total': 0} for level in levels}
    processed_data = []

    for item in tqdm(ds):
        prompt = item['prompt']
        
        for level in levels:
            chosen_key = f'chosen_{level}'
            rejected_key = f'rejected_{level}'
            
            chosen_response = item[chosen_key]
            rejected_response = item[rejected_key]
            
            # Calculate entropy for both responses
            chosen_entropy = generate_entropy(model, tokenizer, chosen_response)
            rejected_entropy = generate_entropy(model, tokenizer, rejected_response)
            
            # Store entropy values
            item[f'chosen_{level}_entropy'] = chosen_entropy
            item[f'rejected_{level}_entropy'] = rejected_entropy
            
            results[f'level_{level}']['total'] += 1
            
            # Lower entropy is better (more confident/certain response)
            if chosen_entropy < rejected_entropy:
                results[f'level_{level}']['correct'] += 1

        processed_data.append(item)

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
    
    with open(filename, 'w') as f:
        json.dump(processed_data, f, indent=2)
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
    with open(filename, 'w') as f:
        json.dump(all_processed_data, f, indent=2)
    print(f"Combined entropy data saved to {filename}")

def main(args):
    login(args.hf_key)
    model, tokenizer = setup_model(args.model_name, args.quantized)
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
        name_match = re.search(r'/([^/]+)$', dataset_name)
        if name_match:
            name = name_match.group(1)
        else:
            name = dataset_name.replace('/', '_')
        
        # Create processed dataset
        processed_dataset = Dataset.from_list(processed_data)
        
        # Save locally instead of pushing to hub (for testing)
        local_path = f"{name}-{args.model_name.split('/')[-1]}-entropy"
        processed_dataset.save_to_disk(local_path)
        print(f"Dataset saved locally to: {local_path}")
        
        # Uncomment below when HF token permissions are fixed
        # processed_dataset.push_to_hub(f"{args.hf_user}/{name}-{args.model_name.split('/')[-1]}-entropy")

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
