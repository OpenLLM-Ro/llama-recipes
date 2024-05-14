import datasets
from .roalpaca_dataset import get_preprocessed_roalpaca_dataset
from .conversations_dataset import get_preprocessed_conversations_dataset
from .rodolly_dataset import get_preprocessed_rodolly_dataset
from .roselfinstruct_dataset import get_preprocessed_roselfinstruct_dataset
from .ronorobots_dataset import get_preprocessed_ronorobots_dataset
from .roorca_dataset import get_preprocessed_roorca_dataset
from .robench_dataset import get_preprocessed_robench_dataset
from .rooasst_dataset import get_preprocessed_rooasst_dataset
from .roultrachat_dataset import get_preprocessed_roultrachat_dataset
from .rocamel_dataset import get_preprocessed_rocamel_dataset

SPLIT = "test"

NPROC = 2

def get_preprocessed_rosft_dataset(dataset_config, tokenizer, split):

    roalpaca_dataset = get_preprocessed_roalpaca_dataset(dataset_config, tokenizer, split, nproc=NPROC)
    rodolly_dataset = get_preprocessed_rodolly_dataset(dataset_config, tokenizer, split, nproc=NPROC)
    roselfinstruct_dataset = get_preprocessed_roselfinstruct_dataset(dataset_config, tokenizer, split, nproc=NPROC)
    ronorobots_dataset = get_preprocessed_ronorobots_dataset(dataset_config, tokenizer, split, nproc=NPROC)
    roorca_dataset = get_preprocessed_roorca_dataset(dataset_config, tokenizer, split, nproc=NPROC)
    robench_dataset = get_preprocessed_robench_dataset(dataset_config, tokenizer, split, nproc=NPROC)
    rooasst_dataset = get_preprocessed_rooasst_dataset(dataset_config, tokenizer, split, nproc=NPROC)
    roultrachat_dataset = get_preprocessed_roultrachat_dataset(dataset_config, tokenizer, split, nproc=NPROC)
    rocamel_dataset = get_preprocessed_rocamel_dataset(dataset_config, tokenizer, split, nproc=NPROC)

    sft_dataset = datasets.concatenate_datasets([roalpaca_dataset, rodolly_dataset, roselfinstruct_dataset, ronorobots_dataset, roorca_dataset, robench_dataset, rooasst_dataset, roultrachat_dataset, rocamel_dataset])
    sft_dataset = sft_dataset.shuffle(seed=42)
    return sft_dataset



if __name__ == "__main__":

    from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=os.getenv("HF_TOKEN"), legacy=False)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[INST]", "[/INST]", "<<SYS>>\n", "\n<</SYS>>\n\n"]})
    # convs_dataset = get_preprocessed_conversations_dataset(None, tokenizer, "dev")
    roalpaca_dataset = get_preprocessed_roalpaca_dataset(None, tokenizer, SPLIT, nproc=NPROC)
    rodolly_dataset = get_preprocessed_rodolly_dataset(None, tokenizer, SPLIT, nproc=NPROC)
    roselfinstruct_dataset = get_preprocessed_roselfinstruct_dataset(None, tokenizer, SPLIT, nproc=NPROC)
    ronorobots_dataset = get_preprocessed_ronorobots_dataset(None, tokenizer, SPLIT, nproc=NPROC)
    roorca_dataset = get_preprocessed_roorca_dataset(None, tokenizer, SPLIT, nproc=NPROC)
    robench_dataset = get_preprocessed_robench_dataset(None, tokenizer, SPLIT, nproc=NPROC)

    # print(convs_dataset)
    print(roalpaca_dataset)
    print(rodolly_dataset)
    print(roselfinstruct_dataset)
    print(ronorobots_dataset)
    print(roorca_dataset)
    print(robench_dataset)
    print()

    sft_dataset = datasets.concatenate_datasets([roalpaca_dataset, rodolly_dataset, roselfinstruct_dataset, ronorobots_dataset, roorca_dataset, robench_dataset])
    print(sft_dataset)

    
