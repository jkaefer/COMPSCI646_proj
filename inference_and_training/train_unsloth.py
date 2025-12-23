from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from unsloth import FastLanguageModel, FastModel

import argparse
import os
import json
import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from accelerate import Accelerator


#new 
from peft import PeftModel
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import gc


os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

def load_dataset(addr, cache_dir):
    def gen():
        with open(addr, 'r') as f:
            dataset = json.load(f)
            for data in dataset:
                yield data
    return datasets.Dataset.from_generator(gen, cache_dir=cache_dir)

parser = argparse.ArgumentParser()

parser.add_argument("--inputs_addr", required=True)
parser.add_argument("--cache_dir", default="/content/drive/MyDrive/mRAG_and_MSRS_source/unsloth_cache")
parser.add_argument("--model_addr", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--per_device_train_batch_size", type=int, default=64)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=0.00005)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--max_steps", type=int, default=5000)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--warmup_steps", type=int, default=250)
parser.add_argument("--max_seq_length", type=int, default=32768)

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = load_dataset(args.inputs_addr, cache_dir=args.cache_dir)

    model, tokenizer = FastModel.from_pretrained(
        model_name = args.model_addr,
        max_seq_length = args.max_seq_length, # Choose any for long context!
        #these are setting weather or not the training should be 4 bit or 8 bit
        #model might already be 4 bit wauntized
        load_in_4bit = True,  # 4 bit quantization to reduce memory
        load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning = False,
        cache_dir = args.cache_dir,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        max_seq_length = args.max_seq_length,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        do_train=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        max_seq_length=args.max_seq_length,
        save_steps=args.save_steps,
        save_only_model=True,
        logging_steps=10,
        # optim="adamw_8bit",
        bf16=True,
        # fp16 = True
    )
    
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
    )
    trainer.train()
    
    #new
    #saving as 4 bit quantized
    print("Merging LoRA into base model...")
    base = AutoModelForCausalLM.from_pretrained(args.model_addr, torch_dtype="float16", device_map="auto")
    merged = PeftModel.from_pretrained(base, args.output_dir)
    merged = merged.merge_and_unload()   # folds LoRA into base
    tok = AutoTokenizer.from_pretrained(args.model_addr, use_fast=True)
    
    #reclaiming some resources
    del base,merged
    gc.collect()
    torch.cuda.empty_cache()
    

    merged_dir = os.path.join(args.output_dir, "_merged_fp16")
    #os.makedirs(merged_dir, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)

    # 2) Prepare GPTQ (small calibration set)
    print("Preparing GPTQ INT4 quantization...")
    qcfg = BaseQuantizeConfig(bits=4,group_size=128,desc_act=False)
    gptq_model = AutoGPTQForCausalLM.from_pretrained(merged_dir, quantize_config=qcfg, device_map="auto")

    # Collect a tiny calibration batch from your training data
    calibration_samples = 100
    n = min(calibration_samples, len(dataset))
    texts = []
    for i in range(n):
        row = dataset[i]
        # try common keys; adjust if your JSON uses a different field name
        text = row.get("text") or row.get("input") or row.get("prompt") or ""
        if not text: continue
        texts.append(text)
    if not texts:
        raise ValueError("No text found for calibration. Ensure your dataset has a 'text'/'input'/'prompt' field.")


    calibration_max_length = 512
    enc = [tok(t, truncation=True, max_length=calibration_max_len) for t in texts]

    print(f"Quantizing with {len(enc)} calibration samples...")
    gptq_model.quantize(enc)  # runs GPTQ fitting

    # 3) Save as GPTQ-INT4
    #os.makedirs(args.gptq_dir, exist_ok=True)
    print(f"Saving GPTQ-INT4 to: /content/drive/MyDrive/mRAG_and_MSRS_source/trained_model")
    gptq_model.save_quantized("/content/drive/MyDrive/mRAG_and_MSRS_source/trained_model", use_safetensors=True)
    tok.save_pretrained("/content/drive/MyDrive/mRAG_and_MSRS_source/trained_model")

    print("Done: GPTQ-INT4 export complete.")
