from datasets import load_dataset, load_from_disk, concatenate_datasets
import numpy as np
import sys
import json
import pyarrow.parquet as pq

def test_cultura():
    # dataset = load_dataset("ft_datasets/cultura_clean/raw", )["train"]
    dataset = load_dataset("ft_datasets/cultura_clean/raw")["train"]
    # dataset = load_dataset("D:\cultura-x-merged")["train"]
    # dataset = load_dataset("D:\cultura-x-merged", data_files=["ro_part_00000.parquet", "ro_part_00001.parquet", "ro_part_00002.parquet"])["train"]
    # print(dataset)

    dataset = dataset.filter(lambda x: len(x["Text"]) > 0 and not(x["Text"].endswith("...")) and len(x["Text"].split(" ")) > 25, num_proc=10)
    dataset = dataset.select_columns(["Text"]).rename_column("Text", "raw_text")
    dataset = dataset.shuffle(seed = 42)
    return dataset
    

    print(dataset)
    sys.exit()
    dataset.save_to_disk("ft_datasets/cultura_clean", max_shard_size="2GB")
    sys.exit()
    # lens_chars = list(map(lambda x: len(x["Text"]), dataset))
    # lens_words = list(map(lambda x: len(x["Text"].split(" ")), dataset))
    # print(len(lens_chars), len(lens_words))

    # for t, x in [("Chars", lens_chars), ("Words", lens_words)]:
    #     print("{0}: Min: {1:3d} Mean: {2:.2f} Median: {3:.2f} Max: {4} Q5: {5:.2f} Q90: {6:2f}".format(t, np.min(x), np.mean(x), np.median(x), np.max(x), np.quantile(x, 0.05), np.quantile(x, 0.9)))
    # sys.exit()


def test_ccnet():

    texts = []
    index = 0
    with open("ft_datasets/cultura_clean/ro_tail.json", "r", encoding="utf-8") as f:
        for line in f:
            if index % 10000 == 0:
                print(index)
            texts.append(eval(line)["raw_content"])
            index += 1
            if index == 50000:
                break
    print(len(texts))
    sys.exit()

    dataset = load_dataset('json', data_files='ft_datasets/cultura_clean/ro_tail.json')["train"]
    print(dataset)

    dataset1 = load_dataset('json', data_files='ft_datasets/cultura_clean/ro_tail_1.json')["train"]
    print(dataset1)
    sys.exit()

    dataset = dataset.filter(lambda x: len(x["raw_content"]) > 0 and len(x["raw_content"].split(" ")) > 25 and x["language"] == "ro", num_proc=10)
    dataset = dataset.select_columns(["raw_content"]).rename_column("raw_content", "raw_text")
    dataset = dataset.shuffle(seed = 42)
    # print(dataset)
    return dataset


    dataset.save_to_disk("ft_datasets/cultura_clean", max_shard_size="2GB")
    sys.exit()

    for x in dataset:
        print(x)
        break


    lens_chars = list(map(lambda x: len(x["raw_content"]), dataset))
    lens_words = list(map(lambda x: len(x["raw_content"].split(" ")), dataset))
    print(len(lens_chars), len(lens_words))

    for t, x in [("Chars", lens_chars), ("Words", lens_words)]:
        print("{0}: Min: {1:3d} Mean: {2:.2f} Median: {3:.2f} Max: {4} Q5: {5:.2f} Q90: {6:2f}".format(t, np.min(x), np.mean(x), np.median(x), np.max(x), np.quantile(x, 0.05), np.quantile(x, 0.9)))
    sys.exit()



if __name__ == "__main__":

    # cultura = test_cultura()
    # print(cultura)
    # sys.exit()
    ccnet = test_ccnet()
    sys.exit()
    print(cultura)

    print(ccnet)
    dataset = concatenate_datasets([cultura, ccnet])
    dataset = dataset.shuffle(seed = 42)
    print(dataset)

    lens_chars = list(map(lambda x: len(x["raw_text"]), dataset))
    lens_words = list(map(lambda x: len(x["raw_text"].split(" ")), dataset))
    print(len(lens_chars), len(lens_words))

    for t, x in [("Chars", lens_chars), ("Words", lens_words)]:
        print("{0}: Min: {1:3d} Mean: {2:.2f} Median: {3:.2f} Max: {4} Q5: {5:.2f} Q90: {6:2f}".format(t, np.min(x), np.mean(x), np.median(x), np.max(x), np.quantile(x, 0.05), np.quantile(x, 0.9)))

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token="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", legacy=False)

    def get_text(sample):
        return {"text": sample["raw_text"]}

    def encode_texts(sample, tokenizer):
        return tokenizer(sample["text"])

    culturaX_dataset = dataset.map(get_text, num_proc=10, remove_columns=["raw_text"], desc="Extract texts")
    culturaX_dataset = culturaX_dataset.map(lambda sample: encode_texts(sample, tokenizer), num_proc=10, batched=True, remove_columns=["text"], desc="Tokenize texts")

    lens = np.array(list(map(lambda x: len(x["input_ids"])+1, culturaX_dataset)))
    print(len(lens))
    print(np.min(lens), np.mean(lens), np.median(lens), np.max(lens), np.quantile(lens, 0.75), np.quantile(lens, 0.85), np.quantile(lens, 0.90))
    for i in [256, 512, 1024, 2048, 4096]:
        print("{0}% over {1}".format(100.0*(lens>i).sum()/len(lens), i))
    print("########################################################################################")
    print()



    # dataset.save_to_disk("ft_datasets/cultura_clean", max_shard_size="1GB")
    # dataset = load_from_disk("ft_datasets/cultura_clean/")
