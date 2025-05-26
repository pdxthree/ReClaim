from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM, TextStreamer
import json
import re
import accelerate
import bitsandbytes
import torch
from Trie import *
from tqdm import tqdm
from peft import PeftModel
import nltk

from nltk import sent_tokenize
import argparse


device = "cuda:0"


def parse_args():
    parser = argparse.ArgumentParser(
        description="generate eval data, define the input and output file path")
    parser.add_argument(
        "--input_path",
        type=str,
        default="eli5_eval_bm25_top100.json",
        help="The path of eval data file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Meta-Llama-3-8B-Instruct",
        help="The base model path",
    )
    parser.add_argument(
        "--claim_path",
        type=str,
        default="sft_epoch5_final",
        help="The base model path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="eli5.json",
        help="The path of output data file",
    )
    parser.add_argument(
        "--docs",
        type=int,
        default=5,
        help="The num of docs",
    )

    args = parser.parse_args()

    return args


def llama3_generate(prompt):
    streamer = TextStreamer(tokenizer, skip_prompt=True,
                            skip_special_tokens=False)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(input_ids, do_sample=True, max_new_tokens=4096, top_k=40, top_p=0.85, temperature=0.2,
                             repetition_penalty=1.1, eos_token_id=128001, bos_token_id=128000, pad_token_id=128257)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):]


def vicuna_generate(prompt):
    streamer = TextStreamer(tokenizer, skip_prompt=True,
                            skip_special_tokens=False)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(input_ids=input_ids, do_sample=True, max_new_tokens=4096, top_k=40, top_p=0.85,
                             temperature=0.2, repetition_penalty=1.0, eos_token_id=2, bos_token_id=1, pad_token_id=32000)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):]


def deal_references(references):
    end_token = "</reference> We can know that: <claim>"
    references_list = references

    references_list.append(end_token)

    references_ids = []
    for reference in references_list:

        reference_ids = tokenizer.encode(
            " " + reference.strip(), return_tensors="pt")[0][1:]
        reference_token_list = reference_ids.tolist()

        references_ids.append(reference_token_list)
    return references_ids


def generate_trie(references):
    sentences = deal_references(references)

    trie = Trie()

    for sentence in sentences:
        trie.insert(sentence)
    return trie


def constrain_generate_fixed_lists_no_lora_with_end_dict(prompt, references, length, repetition_penalty):

    dict_tree_origin = generate_trie(references)
    dict_tree = dict_tree_origin
    root = dict_tree.root
    trie = root
    chosen_tokens = trie.keys()

    next_token = 0
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    output_ids = torch.tensor([[]]).to(device)

    input_len = input_ids.shape[1]
    l = 0
    start_token_list = tokenizer.encode(
        " <reference>", return_tensors="pt").to(device)[0].tolist()[1:]
    start_token_len = len(start_token_list)
    end_token_list = tokenizer.encode(
        " </reference> We can know that: <claim>", return_tensors="pt").to(device)[0].tolist()[1:]

    end_token_len = len(end_token_list)

    Not_allowed_check_token_list = tokenizer.encode(
        " According to the citation: <reference>", return_tensors="pt").to(device)[0].tolist()[1:]
    Not_allowed_check_token_list_len = len(Not_allowed_check_token_list)

    start_check_idx = start_token_len
    end_check_idx = 0

    constrain_flag = False
    past_key_values = None
    integrity_constraint = False

    prev_tokens = []
    inter_prev_tokens = []

    with torch.inference_mode(mode=True):
        while (next_token != 128001 and next_token != 128009 and l < length):
            if start_check_idx == start_token_len:
                constrain_flag = True
                inter_prev_tokens = []
            if end_check_idx == end_token_len:

                constrain_flag = False

                integrity_constraint = True
                inter_prev_tokens = []
                start_check_idx = 0
                end_check_idx = 0

                dict_tree = dict_tree_origin

            logits = model.forward(
                input_ids=input_ids, use_cache=True, past_key_values=past_key_values)
            prob = torch.softmax(logits.logits, dim=-1)[-1][-1]
            for token in prev_tokens:
                prob[token] /= repetition_penalty

            list_sorted = torch.sort(prob, descending=True) # Renamed variable to avoid conflict with built-in list
            pred_next_tokens = list_sorted.indices

            past_key_values = logits.past_key_values

            if -1 in chosen_tokens:

                dict_tree.delete(inter_prev_tokens)

                inter_prev_tokens = []
                root = dict_tree.root
                trie = root
                chosen_tokens = trie.keys()

            if constrain_flag:

                next_token_list = pred_next_tokens.tolist()
                idx = 0

                while not next_token_list[idx] in chosen_tokens:
                    idx += 1
                next_token = pred_next_tokens[idx]

                trie = trie[next_token_list[idx]]
                chosen_tokens = trie.keys()

                inter_prev_tokens.append(next_token_list[idx])

            else:

                flag = False

                next_token = pred_next_tokens[0]

                if integrity_constraint:

                    flag = True

                    if next_token.tolist() == Not_allowed_check_token_list[0]:

                        past_key_values_tmp = past_key_values
                        for i in range(Not_allowed_check_token_list_len - 1):
                            input_ids_tmp = torch.tensor(
                                [[next_token]]).to(device)
                            logits_tmp = model.forward( # Renamed variable to avoid conflict
                                input_ids=input_ids_tmp, use_cache=True, past_key_values=past_key_values_tmp)
                            next_token_val = torch.argmax( # Renamed variable
                                logits_tmp.logits, dim=-1).reshape(-1)[input_ids_tmp.size()[1]-1].tolist()
                            past_key_values_tmp = logits_tmp.past_key_values
                            if next_token_val == Not_allowed_check_token_list[i+1]:
                                next_token = next_token_val # Update next_token if condition met
                                continue
                            else:
                                flag = False
                                break # Exit inner loop if condition not met

                if flag:
                    next_token_list = pred_next_tokens.tolist()
                    idx = 0
                    while idx < len(next_token_list) and next_token_list[idx] == Not_allowed_check_token_list[0]: # Added boundary check for idx
                        idx += 1
                    if idx < len(next_token_list): # Check if a valid token was found
                        next_token = pred_next_tokens[idx]
                    else: # Fallback if all tokens are the not allowed one (should be rare)
                        next_token = pred_next_tokens[0] if len(pred_next_tokens) > 0 else tokenizer.eos_token_id # Fallback or EOS
                else:
                    next_token = pred_next_tokens[0] if len(pred_next_tokens) > 0 else tokenizer.eos_token_id # Fallback or EOS


                integrity_constraint = False
            l += 1

            next_token_ids = torch.tensor([[next_token]]).to(device)

            output_ids = torch.cat((output_ids, next_token_ids), dim=1)

            input_ids = next_token_ids
            if isinstance(next_token, torch.Tensor): # Ensure next_token is a Python int
                next_token = next_token.tolist()


            if start_check_idx >= start_token_len:
                start_check_idx = 0

            if next_token == start_token_list[start_check_idx]:
                start_check_idx += 1
            else:
                start_check_idx = 0 if (start_check_idx > 0 and next_token != start_token_list[0]) else (1 if next_token == start_token_list[0] else 0)


            if end_check_idx >= end_token_len:
                end_check_idx = 0

            if next_token == end_token_list[end_check_idx]:
                end_check_idx += 1
            else:
                end_check_idx = 0 if (end_check_idx > 0 and next_token != end_token_list[0]) else (1 if next_token == end_token_list[0] else 0)


            if l == length: # Corrected variable name from len to length
                print("Reached max length") # Changed message for clarity

            prev_tokens.append(next_token)

        output = tokenizer.decode(output_ids.type(torch.int)[0])

        torch.cuda.empty_cache()

        return output


def generate_prompt(question, references):
    prompt = f"""Given the Question and References below, provide an answer for the Question that is generated using information exclusively from the References(some may be irrelevant). Please use the format of: 'According to the citation: <reference> {{reason1}} </reference> We can know that: <claim> {{answer1}} </claim> According to the citation: <reference> {{reason2}} </reference> We can know that: <claim> {{answer2}} </claim> According to the citation: <reference> {{reason3}} </reference> We can know that ...'. The {{reason}} consists of one or more reference sentences in the References. The {{answer}} is generated based solely on the information contained within {{reason}}. You may employ multiple such structures to organize your answer, ensuring that when all the {{answer}}s are concatenated, they maintain coherence, fluency, and collectively constitute a comprehensive response to the Question. Strive to generate a longer text, utilizing several such structures to organize your response.

# Question:
{question}
# References:
{references}

# Response:
According to the citation: <reference>"""

    return prompt


if __name__ == "__main__":
    args = parse_args()
    
    model = LlamaForCausalLM.from_pretrained(
        args.model_path, device_map=device, torch_dtype=torch.float16, do_sample=False)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, device_map=device, use_fast=False, trust_remote_code=True)

    lora_model = PeftModel.from_pretrained(model, args.claim_path)
    model = lora_model 

    with open(args.input_path, "r") as f:
        old_data = json.load(f)

    print(len(old_data))

    result = []
    err_cnt = 0

    for i in tqdm(range(len(old_data))):
        question = old_data[i]['question']

        references_docs = old_data[i]['docs'][:args.docs] # Renamed to avoid conflict
        references_sentences = []
        reference_text = "" # Renamed to avoid conflict
        for item in references_docs:

            ending_punctuation = {'.', '?', '!', ';'}
            item_sentences = [x.strip() for x in sent_tokenize(item['text'])]
            for item_sentence_idx in range(len(item_sentences)):
                item_sentence = item_sentences[item_sentence_idx].strip()
                reference_text = reference_text + item_sentence + " "
                references_sentences.append(item_sentence)

            reference_text += "\n"

        reference_text = reference_text.rstrip("\n")

        gold_answer = old_data[i]['answer']

        prompt = generate_prompt(question, reference_text)

        # Make sure the global model and tokenizer are accessible by this function
        # This is implicitly handled if 'model' and 'tokenizer' are global variables
        output = constrain_generate_fixed_lists_no_lora_with_end_dict(
            prompt, references_sentences, 1000, 1.0)

        print("According to the citation: <reference>" + output)

        old_data[i].update({
            "question": question,
            "references": references_docs,
            "gold_answer": gold_answer,
            "pred_answer": "According to the citation: <reference>" + output
        })

        result.append(old_data[i])

    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=4)