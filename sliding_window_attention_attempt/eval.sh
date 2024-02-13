python eval.py '/hdd4/zoo/llama2/llama2-7b-chat-hf' 0 1 > original_sampling.log
python eval.py '/hdd4/zoo/llama2/llama2-7b-chat-hf' 1 1 > swa_prefinetune_sampling.log
python eval.py 'llama-2-7b-finetune-swa-5' 1 1 > swa_guanaca_finetune_sampling.log
python eval.py 'llama-2-7b-finetune-swa-billsum-5' 1 1 > swa_billsum_finetune_sampling.log

# python eval.py '/hdd4/zoo/llama2/llama2-7b-chat-hf' 0 0 > original_nosampling.log
# python eval.py '/hdd4/zoo/llama2/llama2-7b-chat-hf' 1 0 > swa_prefinetune_nosampling.log
# python eval.py 'llama-2-7b-finetune-swa-5' 1 0 > swa_guanaca_finetune_nosampling.log
# python eval.py 'llama-2-7b-finetune-swa-billsum-5' 1 0 > swa_billsum_finetune_nosampling.log