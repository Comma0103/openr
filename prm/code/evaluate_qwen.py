from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from datasets import load_dataset
import argparse
import os

from peft import PeftModel,PeftConfig
from peft import get_peft_model, LoraConfig, TaskType
# Ensure bitsandbytes is available for 8-bit quantization
# import bitsandbytes as bnb
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

from torch.nn import BCEWithLogitsLoss
from transformers import DataCollatorWithPadding
from datasets import concatenate_datasets


parser = argparse.ArgumentParser()
# parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
parser.add_argument("--model_path", type=str, default="/home/shaohanh/qilongma/blob/public_models/Qwen2.5-Math-7B-Instruct")
parser.add_argument("--data_path", type=str, default="../../datasets")
parser.add_argument("--per_device_train_batch_size", "--devtrbs", type=int, default=4)
parser.add_argument("--per_device_eval_batch_size", "--devevbs", type=int, default=8)
parser.add_argument("--total_batch_size","--totbs", type=int, default=256)
parser.add_argument("--learning_rate", "--lr", type=float, default=1e-4)
parser.add_argument("--datasets", type=str, default='prm800k')
parser.add_argument("--server", type=str, default='477')
args = parser.parse_args()


good_token = '+'
bad_token = '-'
step_tag = '\n\n\n\n\n' #ки
step_tag2 = '\n\n'


model_path = args.model_path

# tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    add_eos_token=False, 
)

print(tokenizer.encode('a ки b')) # [64, 7665, 1802, 293]
print(tokenizer.encode('a b')) # [64, 293]

print(tokenizer.encode('a \n\n b')) # [64, 4710, 293]
print(tokenizer.encode('a b')) # [64, 293]
print(tokenizer.encode('a \n\n\n\n\n b')) # [64, 76325, 293]
print(tokenizer.encode('a b')) # [64, 293]
print(tokenizer.encode('a \n\n\n\n\n\n b')) # [64, 220, 5134, 293]
print(tokenizer.encode('a b')) # [64, 293]

print(tokenizer.encode('a \n\n\n\n\n\n\n b')) # [64, 22701, 1406, 293]
print(tokenizer.encode('a b')) # [64, 293]

print(tokenizer.encode('a \n\n\n\n\n\n\n\n b')) # [64, 220, 5959, 293]
print(tokenizer.encode('a b')) # [64, 293]

print(tokenizer.encode('a + b')) # [64, 488, 293]
print(tokenizer.encode('a b')) # [64, 293]

print(tokenizer.encode('a - b')) # [64, 481, 293]
print(tokenizer.encode('a b')) # [64, 293]

print(tokenizer.encode(' + -')) # [488, 481]
print(tokenizer.encode('+-')) # [21473]


# if USE_8bit is True:
#     model = prepare_model_for_int8_training(model)
print(tokenizer.eos_token_id) # 151645

tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
tokenizer.padding_side = "left"  # Allow batched inference

# tokenizer = AutoTokenizer.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm')
candidate_tokens = tokenizer.encode(f" {good_token} {bad_token}") # [488, 481]
print(candidate_tokens)
step_tag_id = tokenizer.encode(f" {step_tag}")[-1] # 76325
print(step_tag_id)
print('step_tag_id:',tokenizer.encode(f" {step_tag}"))
print('step_tag_id2:',tokenizer.encode(f"{step_tag2}"))


# model = AutoModelForCausalLM.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm').eval()
# model = AutoModelForCausalLM.from_pretrained(model_path).eval()
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # load_in_8bit=True,   # Enables 8-bit quantization
    device_map="auto",   # Automatically assigns the model to available GPUs/CPUs
    torch_dtype=torch.bfloat16,  # Mixed precision for faster inference
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True,
)
# model.gradient_checkpointing_enable()
# for name,param in model.named_parameters():
#     print(name)
print(model)

# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,  # LoRA for causal language modeling task
#     r=8,  # Rank of LoRA
#     lora_alpha=32,  # Alpha scaling factor for LoRA
#     lora_dropout=0.1,  # Dropout rate for LoRA layers
#     target_modules=["q_proj", "v_proj"],  # Apply LoRA to specific layers
# )
# model = get_peft_model(model, lora_config)

# adapter_path = './prm_results_qwen_new/bs_256_lr_0.0001/checkpoint-6898'
# adapter_path = "/home/shaohanh/qilongma/blob/public_models/Math-psa/checkpoint-2127"
# adapter_path = "/home/shaohanh/qilongma/openr/prm/ckpt/prm_qwen2.5_math_7b_instruct.477/bs_256_lr_0.0001_datasets_prm800k/20241119_043500/checkpoint-20"
adapter_path = "/home/shaohanh/qilongma/blob/test_time_compute_data/ckpt/prm_qwen2.5_math_7b_instruct.gcr/bs_256_lr_0.0001_datasets_all/20241122_065335/checkpoint-4254"
adapter_config = PeftConfig.from_pretrained(adapter_path)

# Wrap the pre-trained model with the LoRA fine-tuned weights
model = PeftModel.from_pretrained(model, adapter_path)
print(adapter_path)

# model.to('cuda:0')
print(model.device)


question = "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
output1 = f"Step 1: Janet's ducks lay 16 eggs per day. {step_tag} Step 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. {step_tag} Step 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. {step_tag} Step 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 {step_tag}" # 18 is right
output2 = f"Step 1: Janet's ducks lay 16 eggs per day. {step_tag} Step 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. {step_tag} Step 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. {step_tag} Step 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 {step_tag}" # 17 is wrong

for output in [output1,output2]:
# for output in [output1, output2,output3]:
    input_for_prm = f"{question} {output}"
    input_id = torch.tensor([tokenizer.encode(input_for_prm)])
    # print(input_id)

    with torch.no_grad():
        logits = model(input_id).logits[:,:,candidate_tokens]
        # print(logits)
        scores = logits.softmax(dim=-1)[:,:,0] # prob of being '+'
        # print(scores)
        step_scores = scores[input_id == step_tag_id]
        
        print(step_scores)
        print('aaaaaa')        
# tensor([0.9609, 0.8359, 0.9258, 0.9258])
# tensor([0.9609, 0.8359, 0.9258, 0.0420])

# exit(0)


def preprocess_function(example):
    input = f"{example['question']} {example['process']}"
    tokenized_inputs = tokenizer(
        input, 
        truncation=True, 
        padding='max_length', 
        # padding=True,
        max_length=2048,
    )
    
    def find_all_indices(lst, element):
        return [i for i, x in enumerate(lst) if x == element]
    
    length = len(tokenized_inputs['input_ids'])
    # print(length)
    indices = find_all_indices(tokenized_inputs['input_ids'],step_tag_id)
    
    if len(indices) != len(example['label']):
        # print(example)
        example['label'] = example['label'][:len(indices)]
    
    assert len(indices) == len(example['label'])
    
    tokenized_inputs['labels'] = [-100] * length
    # tokenized_inputs['attention_mask'] = [1] *length
    # print(len(indices))
    for i in range(len(indices)):
        if example['label'][i] == '+' or example['label'][i] == 1:
            tokenized_inputs['labels'][indices[i]] = candidate_tokens[0]
        elif example['label'][i] == '-' or example['label'][i] == 0:
            tokenized_inputs['labels'][indices[i]] = candidate_tokens[1]
        else:
            raise ValueError('label is wrong')
        tokenized_inputs['attention_mask'][indices[i]] = 0
    # tokenized_inputs['labels'] = [-100] *(length-1) + tokenized_inputs['input_ids'][length-1:]
    
    return tokenized_inputs


DATA_PATH = {
    # "train": 'multi-step.json', 
    # 'train': 'test.json',
    # "test": os.path.join(args.data_path, 'processed_data/prm800k_test.json'), # ori
    # "train": [os.path.join(args.data_path, "processed_data/MATH-APS/math_aps.json"), # ori
    #           os.path.join(args.data_path, 'processed_data/prm800k_train.json')], # ori
    # "train": os.path.join(args.data_path, "processed_data/prm800k/phase2_train.preprocessed.json"),
    # "test": os.path.join(args.data_path, "processed_data/prm800k/phase2_test.preprocessed.json"),
    "train": [os.path.join(args.data_path, "processed_data/prm800k/phase1_train.preprocessed.json")], 
            #   os.path.join(args.data_path, "processed_data/prm800k/phase2_train.preprocessed.json")], 
    "test": [os.path.join(args.data_path, "processed_data/prm800k/phase1_test.preprocessed.json"), 
              os.path.join(args.data_path, "processed_data/prm800k/phase2_test.preprocessed.json")], 
}

dataset = load_dataset('json', data_files=DATA_PATH)

# dataset['train'] = dataset['train'].select(range(10000)) # ori
# dataset['test'] = dataset['test'].select(range(1000))

print('start processing') # tokenize
tokenized_datasets = dataset.map(preprocess_function)
tokenized_datasets['train'] = tokenized_datasets['train'].remove_columns(['question','process','label'])

tokenized_datasets['test'] = tokenized_datasets['test'].remove_columns(['question','process','label'])
print(tokenized_datasets['train'])
print('dataset processed')
# print(tokenized_datasets['train']['input_ids'])
# print(len(tokenized_datasets['train']['input_ids'][0]))

# Data collator for padding inputs dynamically
data_collator = DataCollatorWithPadding(tokenizer)


BATCH_SIZE = args.total_batch_size
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // args.per_device_train_batch_size

world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size
print(world_size)
print(ddp)


output_path = f'{adapter_path}/eval_results'


# Training arguments
training_args = TrainingArguments(
    output_dir=output_path,
    eval_strategy="epoch",  # Evaluate at the end of each epoch
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir=f"{output_path}/logs",
    log_level="info",
    logging_steps=10,
    logging_first_step=True,
    logging_nan_inf_filter=False,
    save_strategy="no",
    save_steps=0.25,
    bf16=True,  # Enable mixed precision for better performance on supported hardware
    report_to="none",  # Set to "wandb" if you are using Weights and Biases for logging
    dataloader_num_workers=4,
    deepspeed=None,
    ddp_find_unused_parameters=False,
)

# Define a custom metric function (e.g., accuracy for binary classification)
def compute_metrics(eval_pred):
    # pass
    # print(eval_pred)
    print('bb')
    pre, labels = eval_pred
    auc = roc_auc_score(pre[1], pre[0])
    ll = log_loss(pre[1], pre[0])
    acc = accuracy_score(pre[1], pre[0] > 0.5)
    result ={
        'auc': auc, 
        'll': ll, 
        'acc': acc, 
    } 
    print(result)
    return result

def preprocess_logits_for_metrics(logits,labels):
    # print('aa')
    # return logits,labels
    labels_index = torch.argwhere(torch.bitwise_or(labels == candidate_tokens[0], labels == candidate_tokens[1]))
    gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == candidate_tokens[1], 0, 1)
    # labels_index[: , 1] = labels_index[: , 1] - 1
    logits = logits[labels_index[:, 0], labels_index[:, 1]][:, [candidate_tokens[1], candidate_tokens[0]]]
    prob = torch.softmax(logits, dim=-1)
    return prob[:, 1], gold
    

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],  # Replace with a validation set if available
    data_collator=data_collator,
    processing_class=tokenizer,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    compute_metrics=compute_metrics,
)

# trainer.train()
trainer.evaluate()

# Save the fine-tuned model and tokenizer
# model.save_pretrained(f'../ckpt/{prm_name}/fine_tuned_math_shepherd_mix_lora_16bit')
# tokenizer.save_pretrained(f'../ckpt/{prm_name}/fine_tuned_math_shepherd_mix_lora_16bit')

# for output in [output1,output2]:
# # for output in [output1, output2,output3]:
#     input_for_prm = f"{question} {output}"
#     input_id = torch.tensor([tokenizer.encode(input_for_prm)])
#     # print(input_id)

#     with torch.no_grad():
#         logits = model(input_id).logits[:,:,candidate_tokens]
#         # print(logits)
#         scores = logits.softmax(dim=-1)[:,:,0] 
#         # print(scores)
#         step_scores = scores[input_id == step_tag_id]
        
#         print(step_scores)
#         print('aaaaaa')        
# # tensor([?, ?, ?, ?])
# # tensor([?, ?, ?, ?])
