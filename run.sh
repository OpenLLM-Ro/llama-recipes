huggingface-cli login --token <HF_TOKEN>
python -m wandb login <WADNB_TOKEN>

#python test.py
#python inference/checkpoint_converter_fsdp_hf.py --fsdp_checkpoint_path test_run_save --consolidated_model_path consolidated_save --HF_model_path_or_name meta-llama/Llama-2-7b-hf
#python inference/checkpoint_converter_fsdp_hf.py --fsdp_checkpoint_path test_run_save --consolidated_model_path consolidated_save --HF_model_path_or_name models/v3/llama7b-full-1e-4_low-chunk1024

#python inference/chat_completion.py --model_name models/v4/llama7b-chat-5e-5_v3_low-chunk1024/ --prompt_file inference/chats.json --do_sample=False --output inference/chats_v4-1e-4-chunk1024
#python inference/chat_completion.py --model_name models/v3/llama7b-full-1e-4_low-chunk1024/ --prompt_file inference/chats.json --do_sample=False --output inference/chats_v3-chunk1024
#python inference/chat_completion.py --model_name models/v2/llama7b-chat_5e-5/ --prompt_file inference/chats.json --do_sample=False --output inference/chats_v2


#python inference/chat_completion.py --model_name org-denis/v3-chunk --prompt_file inference/chats.json --do_sample=False --output inference/chats_v3-chunk-hf


#python inference/chat_completion.py --model_name meta-llama/Llama-2-7b-chat-hf --prompt_file inference/chats.json --peft_model models/denis/ro_alpaca__max_words__2048 --do_sample True --output inference/chats_denis2048_sample
#python inference/chat_completion.py --model_name meta-llama/Llama-2-7b-chat-hf --prompt_file inference/chats.json --peft_model models/denis/ro_alpaca__max_words__2048 --do_sample False --output inference/chats_denis2048
#### multigpu with lora
#
#torchrun --nnodes 1 --nproc_per_node 4  ./llama_finetuning.py --enable_fsdp --model_name meta-llama/Llama-2-7b-hf --use_peft --peft_method lora --pure_bf16 --dataset chat_dataset --output_dir test_save
############ multigpu w/o lora
#
torchrun --nnodes 1 --nproc_per_node 6 ./llama_finetuning.py --enable_fsdp --model_name meta-llama/Llama-2-7b-hf --dist_checkpoint_root_folder test_run_save --dist_checkpoint_folder fine-tuned --pure_bf16 --dataset chat_dataset
#torchrun --nnodes 1 --nproc_per_node 4 ./llama_finetuning.py --enable_fsdp --model_name models/v3/llama7b-full-1e-4_low-chunk1024 --dist_checkpoint_root_folder test_run_save --dist_checkpoint_folder fine-tuned --pure_bf16 --dataset chat_dataset
#
#################### bbbbbbbbbbbbbbbbbbbbbbbbbbb #################### 
############ test something
#
#python ./ft_datasets/chat_dataset.py
