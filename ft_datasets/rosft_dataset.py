import datasets
from roalpaca_dataset import get_preprocessed_roalpaca_dataset
from conversations_dataset import get_preprocessed_conversations_dataset
from rodolly_dataset import get_preprocessed_rodolly_dataset
from roselfinstruct_dataset import get_preprocessed_roselfinstruct_dataset
from ronorobots_dataset import get_preprocessed_ronorobots_dataset

SPLIT = "test"

def get_preprocessed_rosft_dataset(dataset_config, tokenizer, split):

    roalpaca_dataset = get_preprocessed_roalpaca_dataset(dataset_config, tokenizer, split)
    rodolly_dataset = get_preprocessed_rodolly_dataset(dataset_config, tokenizer, split)
    roselfinstruct_dataset = get_preprocessed_roselfinstruct_dataset(dataset_config, tokenizer, split)
    ronorobots_dataset = get_preprocessed_ronorobots_dataset(dataset_config, tokenizer, split)
    
    sft_dataset = datasets.concatenate_datasets([roalpaca_dataset, rodolly_dataset, roselfinstruct_dataset, ronorobots_dataset])
    sft_dataset = sft_dataset.shuffle(seed=42)
    return sft_dataset



if __name__ == "__main__":

    from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token="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", legacy=False)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[INST]", "[/INST]", "<<SYS>>\n", "\n<</SYS>>\n\n"]})
    # convs_dataset = get_preprocessed_conversations_dataset(None, tokenizer, "dev")
    roalpaca_dataset = get_preprocessed_roalpaca_dataset(None, tokenizer, SPLIT)
    rodolly_dataset = get_preprocessed_rodolly_dataset(None, tokenizer, SPLIT)
    roselfinstruct_dataset = get_preprocessed_roselfinstruct_dataset(None, tokenizer, SPLIT)
    ronorobots_dataset = get_preprocessed_ronorobots_dataset(None, tokenizer, SPLIT)
    
    # print(convs_dataset)
    print(roalpaca_dataset)
    print(rodolly_dataset)
    print(roselfinstruct_dataset)
    print(ronorobots_dataset)

    sft_dataset = datasets.concatenate_datasets([roalpaca_dataset, rodolly_dataset, roselfinstruct_dataset, ronorobots_dataset])
    print(sft_dataset)

    