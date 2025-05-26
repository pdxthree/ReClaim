from transformers import (
    AutoModel,
    AutoTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    TextStreamer,
)
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

device = "cuda:3"

def parse_args():
    parser = argparse.ArgumentParser(
        description="generate eval data, define the input and output file path"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="eli5_eval_bm25_top100.json",
        help="The path of eval data file",
    )
    parser.add_argument(
        "--citation_model",
        type=str,
        default="sft",
        help="The base model path",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Meta-Llama-3-8B-Instruct",
        help="The base model path",
    )
    parser.add_argument(
        "--claim_model",
        type=str,
        default="sft_claim",
        help="The base model path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="eli5_result.json",
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


def deal_references(references):
    end_token = "</reference>"
    references_list = references  

    references_list.append(end_token)
    references_ids = []
    for reference in references_list:
        reference_ids = tokenizer.encode(
            " " + reference.strip(), return_tensors="pt"
        ).to(device)[0][1:]
        reference_token_list = reference_ids.tolist()

        references_ids.append(reference_token_list)
    return references_ids


def generate_trie(references):
    sentences = deal_references(references)
    trie = Trie()
    for sentence in sentences:
        trie.insert(sentence)
    return trie


def constrain_generate_fixed_lists_no_lora_with_end_dict(
    prompt, question, references, length, repetition_penalty
):
    model.set_adapter("citation_generate")
    dict_tree_origin = generate_trie(references)
    dict_tree = dict_tree_origin
    root = dict_tree.root
    trie = root
    chosen_tokens = trie.keys()

    next_token = 0
    question_ids = tokenizer.encode(question, return_tensors="pt").to(device)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    origin_input_ids = input_ids.clone()
    output_ids = torch.tensor([[]]).to(device)

    input_len = len(origin_input_ids[0].tolist())
    l = 0

    start_token_list = (
        tokenizer.encode(" <reference>", return_tensors="pt").to(
            device)[0].tolist()[1:]
    )
    prefix_ids = (
        tokenizer.encode(" According to the citation:",
                         return_tensors="pt").to(device)
    )[:, 1:]
    prefix_ids_claim = tokenizer.encode(
        " We can know that:", return_tensors="pt").to(device)[:, 1:]
    start_token_len = len(start_token_list)

    end_token_list = (
        tokenizer.encode(" </reference>", return_tensors="pt")
        .to(device)[0]
        .tolist()[1:]
    )
    end_token_len = len(end_token_list)

    lora_change_token = (
        tokenizer.encode(
            " </claim>", return_tensors="pt").to(device)[0].tolist()[1:]
    )
    lora_change_token_len = len(lora_change_token)

    start_check_idx = 0
    end_check_idx = 0
    lora_check_idx = 0

    constrain_flag = False
    past_key_values = None
    claim_generate_flag = False
    first_step_citation = False
    first_step_claim = False

    prev_tokens = []
    inter_prev_tokens = []

    with torch.inference_mode(mode=True):
        reference_cnt = 0
        while ((next_token != 128001 and next_token != 128009) or reference_cnt < 2) and l < length:
            if start_check_idx == start_token_len:
                constrain_flag = True
                inter_prev_tokens = []
            if end_check_idx == end_token_len:
                constrain_flag = False
                inter_prev_tokens = []

                print("-----claim_generate-----")

                claim_generate_flag = True
                first_step_claim = True
                model.set_adapter("claim_generate")

                start_check_idx = 0
                end_check_idx = 0
                dict_tree = dict_tree_origin
            if lora_check_idx == lora_change_token_len:
                reference_cnt += 1
                if reference_cnt == 6:
                    break
                
                print("-----citation_generate-----")
                model.set_adapter("citation_generate")
                claim_generate_flag = False
                first_step_citation = True
                lora_check_idx = 0

            if claim_generate_flag:
                if first_step_claim:
                    past_key_values = None  
                    
                    input_ids = output_ids.clone().type(torch.int)
                    first_step_claim = False
                logits = model.forward(
                    input_ids=input_ids, use_cache=True, past_key_values=past_key_values
                )
            else:
                if first_step_citation:
                    past_key_values = None
                    if reference_cnt < 2:
                        output_ids = torch.cat(
                            (output_ids.type(torch.int), prefix_ids), dim=1).type(torch.int)
                        input_ids = torch.cat((origin_input_ids, output_ids), dim=1).type(
                            torch.int
                        )
                    else:
                        input_ids = torch.cat((origin_input_ids, output_ids), dim=1).type(
                            torch.int
                        )
                    logits = model.forward(
                        input_ids=input_ids, use_cache=True, past_key_values=past_key_values
                    )  
                    prob = torch.softmax(logits.logits, dim=-1)[-1][-1]
                    list = torch.sort(prob, descending=True)
                    pred_next_tokens = list.indices
                    next_token = pred_next_tokens[0]
                    next_word = tokenizer.decode(next_token)
                    print(f"next_token:{next_token}", f"next_word:{next_word}", f"当前生成长度:{l}")
                    next_token = next_token.tolist()
                    if next_token == 128009:
                        output = tokenizer.decode(output_ids.type(torch.int)[0])
                        print("end")
                        return output
                    first_step_citation = False
                logits = model.forward(
                    input_ids=input_ids, use_cache=True, past_key_values=past_key_values
                )
            prob = torch.softmax(logits.logits, dim=-1)[-1][-1]
            for token in prev_tokens:
                prob[token] /= repetition_penalty

            list = torch.sort(prob, descending=True)
            pred_next_tokens = list.indices

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
                next_token = pred_next_tokens[0]

            l += 1

            next_token_ids = torch.tensor([[next_token]]).to(device)
            output_ids = torch.cat((output_ids, next_token_ids), dim=1)

            input_ids = next_token_ids
            next_token = next_token.tolist()

            if start_check_idx >= start_token_len:
                start_check_idx = 0

            if next_token == start_token_list[start_check_idx]:
                start_check_idx += 1
            else:
                start_check_idx = 0 if next_token != start_token_list[0] else 1

            if end_check_idx >= end_token_len:
                end_check_idx = 0

            if next_token == end_token_list[end_check_idx]:
                end_check_idx += 1
            else:
                end_check_idx = 0 if next_token != end_token_list[0] else 1

            if next_token == lora_change_token[lora_check_idx]:
                lora_check_idx += 1
            else:
                lora_check_idx = 0 if next_token != lora_change_token[0] else 1

            if l == len:
                print("err")

            prev_tokens.append(next_token) 

            if next_token == 128001 or (claim_generate_flag == False and next_token == 128009):
                break

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
"""

    return prompt

if __name__ == "__main__":
    args = parse_args()
    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model, device_map=device, torch_dtype=torch.float16, do_sample=False
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, device_map=device, use_fast=False, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(
        base_model, args.citation_model, adapter_name="citation_generate")

    model.load_adapter(args.claim_model, adapter_name="claim_generate")
    model.set_adapter("claim_generate")

    with open(args.input_path, "r") as f:
        old_data = json.load(f)

    print(len(old_data))

    result = []
    err_cnt = 0

    for i in tqdm(range(len(old_data))):
        question = old_data[i]["question"]
        references = old_data[i]["docs"][:5]
        references_sentences = []
        reference = ""
        for item in references:
            ending_punctuation = {".", "?", "!", ";"}
            item_sentences = [x.strip() for x in sent_tokenize(item["text"])]
            for item_sentence_idx in range(len(item_sentences)):
                item_sentence = item_sentences[item_sentence_idx].strip()
                
                reference = reference + item_sentence + " "
                references_sentences.append(item_sentence)
            reference += "\n"
        reference = reference.rstrip("\n")

        gold_answer = old_data[i]["answer"]

        prompt = generate_prompt(question, reference)
        output = constrain_generate_fixed_lists_no_lora_with_end_dict(
            prompt, question, references_sentences, 1000, 1.0
        )
        print(output)

        old_data[i].update(
            {
                "question": question,
                "references": references,
                "gold_answer": gold_answer,
                "pred_answer": output,
            }
        )

        result.append(old_data[i])

    with open(args.output_path, "w") as f:
        json.dump(result, f)