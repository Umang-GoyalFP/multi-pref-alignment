def generate_sequence_entropy(model, tokenizer, text):
    """Calculate entropy across the entire sequence, not just last token"""
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(model.device)
    
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

def generate_perplexity(model, tokenizer, text):
    """Alternative: Use perplexity instead of entropy"""
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    
    return perplexity

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
            
            # Calculate both entropy and perplexity for comparison
            chosen_entropy = generate_sequence_entropy(model, tokenizer, chosen_response)
            rejected_entropy = generate_sequence_entropy(model, tokenizer, rejected_response)
            
            chosen_perplexity = generate_perplexity(model, tokenizer, chosen_response)
            rejected_perplexity = generate_perplexity(model, tokenizer, rejected_response)
            
            # Store all metrics
            item[f'chosen_{level}_entropy'] = chosen_entropy
            item[f'rejected_{level}_entropy'] = rejected_entropy
            item[f'chosen_{level}_perplexity'] = chosen_perplexity
            item[f'rejected_{level}_perplexity'] = rejected_perplexity
            
            results[f'level_{level}']['total'] += 1
            
            # DECISION: Which metric to use for "correctness"?
            # Option 1: Lower entropy (more confident)
            entropy_prefers_chosen = chosen_entropy < rejected_entropy
            
            # Option 2: Lower perplexity (more likely under model)
            perplexity_prefers_chosen = chosen_perplexity < rejected_perplexity
            
            # You could use either or combine them
            if entropy_prefers_chosen:  # or perplexity_prefers_chosen
                results[f'level_{level}']['correct'] += 1

        processed_data.append(item)

    accuracies = {
        level: (results[level]['correct'] / results[level]['total']) * 100 
        if results[level]['total'] > 0 else 0 
        for level in results
    }

    return accuracies, processed_data

def save_all_accuracies_to_json(all_accuracies, model_name):
    """Save accuracies to JSON file"""
    filename = f"accuracy-rm-bench-{model_name.split('/')[-1]}-entropy.json"
    with open(filename, 'w') as f:
        json.dump(all_accuracies, f, indent=4)
    print(f"All accuracies saved to {filename}")

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
    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)['train']
        accuracies, processed_data = evaluate_rewards(dataset, model, tokenizer, dataset_name)
        
        all_accuracies[dataset_name] = accuracies  
        for level, acc in accuracies.items():
            print(f"Accuracy for {dataset_name} - {level}: {acc:.2f}%")
        
        name = re.search(r'/([^/]+)
, dataset_name).group(1)
        processed_dataset = Dataset.from_list(processed_data)
        processed_dataset.push_to_hub(f"{args.hf_user}/{name}-{args.model_name.split('/')[-1]}-entropy")

    save_all_accuracies_to_json(all_accuracies, args.model_name)
    del model
    gc.collect()
