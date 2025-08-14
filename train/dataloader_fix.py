def get_wmt(split: str, n_examples: Optional[int] = None) -> Dataset:
    rank0_print(f'Loading WMT dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset("wmt/wmt19", "de-en", split=split)

    data = Dataset("wmt")

    prefix_prompt = "Translate the following English sentences to German."

    few_shot_example_path = "./src/mydatasets/wmt/prompt_fewshot.txt"

    def attach_template(dataset, template_path=few_shot_example_path, n_examples=n_examples):
        template = Template(read_txt(template_path))

        result = []
        count = 0

        for data in dataset:
            data=data['translation']
            answer = {"role": "assistant", "content": data['de']}

            result.append({
                'prompt': template.render(**data),
                'reference': answer,
                'english': data['en']
            })
            count += 1
            if count == n_examples:
                break
        
        return result
    
    dataset = attach_template(dataset)
    
    for row in dataset:
        key = row['english']
        # Fix: Format the prompt as a list of dictionaries with "role" and "content" keys
        data[key].prompt = [{"role": "user", "content": row['prompt']}]
        data[key].original_prompt = [row['prompt']]
        data[key].generations = [row['reference']]  # This is already in the correct format with role and content
        data[key].question = [row['english']]
        data[key].sft_index = 0
        data[key].dataset_name = data.name
        data[key].truncation_mode = 'target'

    return data
