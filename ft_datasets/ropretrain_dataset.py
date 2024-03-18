from datasets import load_dataset, load_from_disk, concatenate_datasets
import numpy as np
import sys
import json
import pyarrow.parquet as pq
import os


def test_cultura():
    dataset = load_dataset("ft_datasets/cultura_clean/raw")["train"]
    #dataset = load_dataset("ft_datasets/cultura_clean/raw", data_files="ro_part_00000.parquet")["train"]
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

    texts = set()
    index = 0
    folder = "ft_datasets/ccnet/raw/2019-26"
    
    for file in os.listdir(folder):
        index = 0
        file_path = os.path.join(folder, file)
        if not (file_path.endswith(".json")):
            continue
        print(file)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if index % 500000 == 0:
                    print(index, len(texts), flush=True)

                e = eval(line)
                if e["language"] != "ro":
                    continue

                texts.add(e["raw_content"])
                index += 1
        print("Done {0}. Crt size {1}".format(file, len(texts)), flush=True)
        #print(len(texts))
        #break

    print("Done full. Crt size: {0}".format(len(texts)))


    save_index = 84
    split = 500000
    crt_index = 0
    ds = []
    for text in texts:
        d = {}
        d["raw_content"] = text
        ds.append(d)
        crt_index += 1
        if crt_index >= split:
            print("Saving index: {0}. Saving data: {1}".format(save_index, len(ds)), flush=True)
            json.dump(ds, open(os.path.join(folder, "out_{0:03d}.json".format(save_index)), "w", encoding="utf-8"))
            save_index += 1
            crt_index = 0
            ds = []
    
    print("Saving index: {0}. Saving data: {1}".format(save_index, len(ds)), flush=True)
    json.dump(ds, open(os.path.join(folder, "out_{0:03d}.json".format(save_index)), "w", encoding="utf-8"))
    sys.exit()

    dataset = load_dataset('json', data_files='ft_datasets/ccnet/raw/dedups/out_8.json')["train"]
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



def load_ccnet():
    dataset = load_dataset('json', data_files='ft_datasets/ccnet/raw/dedups/*.json')["train"]
    #dataset = load_dataset('json', data_files='ft_datasets/ccnet/raw/dedups/out_000.json')["train"]
    dataset = dataset.filter(lambda x: len(x["raw_content"]) > 0 and len(x["raw_content"].split(" ")) > 25, num_proc=10)
    dataset = dataset.select_columns(["raw_content"]).rename_column("raw_content", "raw_text")
    dataset = dataset.shuffle(seed = 42)
    #print(dataset)
    #sys.exit()
    return dataset


if __name__ == "__main__":
    cultura = test_cultura()
    #ccnet = test_ccnet()
    ccnet = load_ccnet()
    print("Cultura:", cultura, flush=True)
    print("CCNet:", ccnet, flush=True)
    dataset = concatenate_datasets([cultura, ccnet])
    dataset = dataset.shuffle(seed = 42)
    print(dataset)

    dataset.save_to_disk("ft_datasets/ccnet_cultura", max_shard_size="1GB")
    print("Saved", flush=True)


    lens_chars = list(map(lambda x: len(x["raw_text"]), dataset))
    lens_words = list(map(lambda x: len(x["raw_text"].split(" ")), dataset))
    print(len(lens_chars), len(lens_words), flush=True)

    for t, x in [("Chars", lens_chars), ("Words", lens_words)]:
        print("{0}: Min: {1:3d} Mean: {2:.2f} Median: {3:.2f} Max: {4} Q5: {5:.2f} Q90: {6:2f}".format(t, np.min(x), np.mean(x), np.median(x), np.max(x), np.quantile(x, 0.05), np.quantile(x, 0.9)))

    #from transformers import AutoTokenizer, AutoModelForCausalLM
    #tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token="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", legacy=False)

    #def get_text(sample):
    #    return {"text": sample["raw_text"]}

    #def encode_texts(sample, tokenizer):
    #    return tokenizer(sample["text"])

    #culturaX_dataset = dataset.map(get_text, num_proc=10, remove_columns=["raw_text"], desc="Extract texts")
    #culturaX_dataset = culturaX_dataset.map(lambda sample: encode_texts(sample, tokenizer), num_proc=10, batched=True, remove_columns=["text"], desc="Tokenize texts")

    #lens = np.array(list(map(lambda x: len(x["input_ids"])+1, culturaX_dataset)))
    #print(len(lens))
    #print(np.min(lens), np.mean(lens), np.median(lens), np.max(lens), np.quantile(lens, 0.75), np.quantile(lens, 0.85), np.quantile(lens, 0.90))
    #for i in [256, 512, 1024, 2048, 4096]:
    #    print("{0}% over {1}".format(100.0*(lens>i).sum()/len(lens), i))
    #print("########################################################################################")
    #print()

    dataset = load_from_disk("ft_datasets/ccnet_cultura/")
    print("Loaded:", dataset)
