import sys
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM, TextStreamer
import json
import re
import accelerate
import bitsandbytes
import torch
import argparse
from Trie import *
from tqdm import tqdm
from peft import PeftModel
from nltk import sent_tokenize

device = "cuda:0"


def parse_args():
    parser = argparse.ArgumentParser(
        description="generate eval data, define the input and output file path")
    parser.add_argument(
        "--input_path",
        type=str,
        default="expertqa_reranked.json",
        help="The path of eval data file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Meta-Llama-3-8B-Instruct",
        help="The base model path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="expert_3shot.json",
        help="The path of output data file",
    )
    parser.add_argument(
        "--docs",
        type=int,
        default=5,
        help="The num of docs",
    )
    parser.add_argument(
        "--shot",
        type=int,
        default=3,
        help="The num of shot",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="The device id",
    )
    args = parser.parse_args()
    return args


def llama3_generate(prompt):
    messages = [
        {"role": "system", "content": "Follow my instructions to complete my task. Give me an response and don't generate other irrelevant information"},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.01,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=False)


def vicuna_generate(prompt):
    streamer = TextStreamer(tokenizer, skip_prompt=True,
                            skip_special_tokens=False)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(input_ids, do_sample=True, max_new_tokens=4096, top_k=40, top_p=0.85,
                             temperature=0.2, repetition_penalty=1.1, eos_token_id=2, bos_token_id=1, pad_token_id=0)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):]


def generate_direct_prompt_shot2(question, reference):
    instruction = "Given the Question and References below, provide an answer for the Question that is generated using information exclusively from the References(some may be irrelevant). Please use the format of: 'According to the citation: <reference> {reason1} </reference> We can know that: <claim> {answer1} </claim> According to the citation: <reference> {reason2} </reference> We can know that: <claim> {answer2} </claim> According to the citation: <reference> {reason3} </reference> We can know that ...'. The {reason} consists of one or more reference sentences in the References. The {answer} is generated based solely on the information contained within {reason}. You may employ multiple such structures to organize your answer, ensuring that when all the {answer}s are concatenated, they maintain coherence, fluency, and collectively constitute a comprehensive response to the Question. Strive to generate a longer text, utilizing several such structures to organize your response."
    prompt = """{}

Here are some examples:
### Input:
# Question: Why are snowflakes usually six-sided?
# References: To add to both Spießbürger's and casey's excellent answers, hydrogen bonds are the reason why some snowflakes are six-sided. His was alluded to, but I think it could use a bit more extrapolation.
Though snowflakes are beautifully varied, there is one underlying pattern that is seldom broken: snowflakes’ intricate patterns (almost) always have six sides. The reason why, says science blogger Megan Nantel, is because snowflakes are made of water, and water molecules bonded together take on particular shapes.
Some people would have learned, or realized, that every snowflake has six sides, regardless of its shape or size. If you didn’t know that already, how cool is that?
A seasonal banner in Brooklyn, New York features impossible, eight-sided snowflakes. Benedict explains that snowflakes have six sides because they are ice crystals, which consist of water molecules arranged in a lattice of hexagonal rings. Credit: Christopher L. Cahill
So we’ve uncovered the reason for all snowflakes having six sides. But why then, if given the opportunity to expand in all directions, do snowflakes grow so thin? Why do their corners grow many thousand times faster than the top and bottom faces?

### Response:
According to the citation: <reference> Though snowflakes are beautifully varied, there is one underlying pattern that is seldom broken: snowflakes’ intricate patterns (almost) always have six sides. The reason why, says science blogger Megan Nantel, is because snowflakes are made of water, and water molecules bonded together take on particular shapes. </reference> We can know that: <claim> Snowflakes usually have six sides because they are made of water molecules bonded together, which take on particular shapes. </claim> According to the citation: <reference> Benedict explains that snowflakes have six sides because they are ice crystals, which consist of water molecules arranged in a lattice of hexagonal rings. </reference> We can know that: <claim> Water molecules form a lattice of hexagonal rings, which is why snowflakes have six sides regardless of their shape or size. </claim> According to the citation: <reference> So we’ve uncovered the reason for all snowflakes having six sides. But why then, if given the opportunity to expand in all directions, do snowflakes grow so thin? Why do their corners grow many thousand times faster than the top and bottom faces? </reference> We can know that: <claim> However, when given the opportunity to expand in all directions, the corners of snowflakes grow much faster than the top and bottom faces, resulting in their thin and intricate patterns. </claim>

-

### Input:
# Question: Why do your pupils dilate when you see someone you are attracted to?
# References: So, for those of you who are curious, pupils do dilate when we are attracted to someone. In fact, pupils can dilate by up to 45% when we look at someone we love, aahhhh…
Though a person’s dilated pupil can be a tell-tale sign of attraction, it’s a pretty subtle clue. Many other creatures seem to express their willingness to mate much more openly.
This may be because we are hardwired to understand that dilated pupils equate to attraction, so the men were more likely to find attractive the women whose eyes were signaling that they were also feeling that attraction.
Research has found that heterosexual men are more attracted to women when their pupils are dilated. Being shown two photos of the same woman where the size of pupils have been altered to be different in each, men found women with larger pupils to be more attractive and open. The men described the woman with bigger pupils as “soft”, “pretty” and “feminine”, while characterising the woman with the reduced pupils as “hard”, “cold” and “selfish”. It should be noted that none of the men noticed the detail change in the photos.
This pupil dilation may in turn play a part in who we pursue romantically, according to a 1975 study that found that men found women with larger pupils more attractive. The men in the study described women with larger pupils as more feminine and soft, while women with smaller pupils were considered cold or hard.

### Response:
According to the citation: <reference> This may be because we are hardwired to understand that dilated pupils equate to attraction, so the men were more likely to find attractive the women whose eyes were signaling that they were also feeling that attraction. </reference> We can know that: <claim> Pupils dilate when you see someone you are attracted to because we are hardwired to understand that dilated pupils equate to attraction. </claim> According to the citation: <reference> Research has found that heterosexual men are more attracted to women when their pupils are dilated. Being shown two photos of the same woman where the size of pupils have been altered to be different in each, men found women with larger pupils to be more attractive and open. </reference> We can know that: <claim> Research has found that heterosexual men are more attracted to women when their pupils are dilated. </claim> According to the citation: <reference> This pupil dilation may in turn play a part in who we pursue romantically, according to a 1975 study that found that men found women with larger pupils more attractive. </reference> We can know that: <claim> Pupil dilation may play a part in who we pursue romantically. </claim>

Now here is a case need your help:
### Input:
# Question: {}
# References: {}

### Response:
According to the citation: <reference>""".format(instruction, question, reference)
    return prompt


def generate_direct_prompt_shot3(question, reference):
    instruction = "Given the Question and References below, provide an answer for the Question that is generated using information exclusively from the References(some may be irrelevant). Please use the format of: 'According to the citation: <reference> {reason1} </reference> We can know that: <claim> {answer1} </claim> According to the citation: <reference> {reason2} </reference> We can know that: <claim> {answer2} </claim> According to the citation: <reference> {reason3} </reference> We can know that ...'. The {reason} consists of one or more reference sentences in the References. The {answer} is generated based solely on the information contained within {reason}. You may employ multiple such structures to organize your answer, ensuring that when all the {answer}s are concatenated, they maintain coherence, fluency, and collectively constitute a comprehensive response to the Question. Strive to generate a longer text, utilizing several such structures to organize your response."
    prompt = """{}

Here are some examples:
### Input:
# Question: Why are snowflakes usually six-sided?
# References: To add to both Spießbürger's and casey's excellent answers, hydrogen bonds are the reason why some snowflakes are six-sided. His was alluded to, but I think it could use a bit more extrapolation.
Though snowflakes are beautifully varied, there is one underlying pattern that is seldom broken: snowflakes’ intricate patterns (almost) always have six sides. The reason why, says science blogger Megan Nantel, is because snowflakes are made of water, and water molecules bonded together take on particular shapes.
Some people would have learned, or realized, that every snowflake has six sides, regardless of its shape or size. If you didn’t know that already, how cool is that?
A seasonal banner in Brooklyn, New York features impossible, eight-sided snowflakes. Benedict explains that snowflakes have six sides because they are ice crystals, which consist of water molecules arranged in a lattice of hexagonal rings. Credit: Christopher L. Cahill
So we’ve uncovered the reason for all snowflakes having six sides. But why then, if given the opportunity to expand in all directions, do snowflakes grow so thin? Why do their corners grow many thousand times faster than the top and bottom faces?

### Response:
<reference> Though snowflakes are beautifully varied, there is one underlying pattern that is seldom broken: snowflakes’ intricate patterns (almost) always have six sides. The reason why, says science blogger Megan Nantel, is because snowflakes are made of water, and water molecules bonded together take on particular shapes. </reference> <claim> Snowflakes usually have six sides because they are made of water molecules bonded together, which take on particular shapes. </claim> <reference> Benedict explains that snowflakes have six sides because they are ice crystals, which consist of water molecules arranged in a lattice of hexagonal rings. </reference> <claim> Water molecules form a lattice of hexagonal rings, which is why snowflakes have six sides regardless of their shape or size. </claim> <reference> So we’ve uncovered the reason for all snowflakes having six sides. But why then, if given the opportunity to expand in all directions, do snowflakes grow so thin? Why do their corners grow many thousand times faster than the top and bottom faces? </reference> <claim> However, when given the opportunity to expand in all directions, the corners of snowflakes grow much faster than the top and bottom faces, resulting in their thin and intricate patterns. </claim>

-

### Input:
# Question: Why do your pupils dilate when you see someone you are attracted to?
# References: So, for those of you who are curious, pupils do dilate when we are attracted to someone. In fact, pupils can dilate by up to 45% when we look at someone we love, aahhhh…
Though a person’s dilated pupil can be a tell-tale sign of attraction, it’s a pretty subtle clue. Many other creatures seem to express their willingness to mate much more openly.
This may be because we are hardwired to understand that dilated pupils equate to attraction, so the men were more likely to find attractive the women whose eyes were signaling that they were also feeling that attraction.
Research has found that heterosexual men are more attracted to women when their pupils are dilated. Being shown two photos of the same woman where the size of pupils have been altered to be different in each, men found women with larger pupils to be more attractive and open. The men described the woman with bigger pupils as “soft”, “pretty” and “feminine”, while characterising the woman with the reduced pupils as “hard”, “cold” and “selfish”. It should be noted that none of the men noticed the detail change in the photos.
This pupil dilation may in turn play a part in who we pursue romantically, according to a 1975 study that found that men found women with larger pupils more attractive. The men in the study described women with larger pupils as more feminine and soft, while women with smaller pupils were considered cold or hard.

### Response:
<reference> This may be because we are hardwired to understand that dilated pupils equate to attraction, so the men were more likely to find attractive the women whose eyes were signaling that they were also feeling that attraction. </reference> <claim> Pupils dilate when you see someone you are attracted to because we are hardwired to understand that dilated pupils equate to attraction. </claim> <reference> Research has found that heterosexual men are more attracted to women when their pupils are dilated. Being shown two photos of the same woman where the size of pupils have been altered to be different in each, men found women with larger pupils to be more attractive and open. </reference> <claim> Research has found that heterosexual men are more attracted to women when their pupils are dilated. </claim> <reference> This pupil dilation may in turn play a part in who we pursue romantically, according to a 1975 study that found that men found women with larger pupils more attractive. </reference> <claim> Pupil dilation may play a part in who we pursue romantically. </claim>

-

### Input:
# Question: What is the difference between an iMessage and a Text Message in iOS 5?
# References: - iMessage: An iMessage is a type of text message that can only be sent from an Apple device to another Apple device, like an iPhone or an iPad. An iMessage can include standard text like an SMS message, but also photos, videos, and other add-ons like GIFs. iMessages require an internet connection (either through Wi-Fi or a data plan) and are automatically encrypted.
- The most basic difference between an iMessage and a text message is that an iMessage is sent using an internet connection and the latter is delivered over a cellular network.
- Another difference is that the iMessage service can be accessed only on Apple devices and the SMS service can be used on any device.
iMessages are texts, photos, or videos that you send to another iPhone, iPad, iPod touch, or Mac over Wi-Fi or cellular-data networks. These messages are always encrypted and appear in blue text bubbles. To turn iMessage on or off, go to Settings > Messages.
You are here: Home / Apple Services / What is iMessage and how is it different to normal text messages?

### Response:
<reference> The most basic difference between an iMessage and a text message is that an iMessage is sent using an internet connection and the latter is delivered over a cellular network. </reference> <claim> The main difference between an iMessage and a Text Message in iOS 5 is that an iMessage is sent using an internet connection, while a text message is sent over a cellular network. </claim> <reference> iMessage: An iMessage is a type of text message that can only be sent from an Apple device to another Apple device, like an iPhone or an iPad. An iMessage can include standard text like an SMS message, but also photos, videos, and other add-ons like GIFs. </reference> <claim> iMessages are only accessible on Apple devices, and can include standard text, photos, videos and other add-ons like GIFs. </claim> <reference> iMessages are texts, photos, or videos that you send to another iPhone, iPad, iPod touch, or Mac over Wi-Fi or cellular-data networks. These messages are always encrypted and appear in blue text bubbles. </reference> <claim> They are encrypted and appear in blue text bubbles. </claim>

Now here is a case need your help:
### Input:
# Question: {}
# References: {}

### Response:
""".format(instruction, question, reference)
    return prompt


if __name__ == "__main__":
    args = parse_args()
    device = args.device
    model = LlamaForCausalLM.from_pretrained(
        args.model_path, device_map=device, torch_dtype=torch.float16, do_sample=False)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, device_map=device, use_fast=False, trust_remote_code=True)

    with open(args.input_path, "r") as f:
        old_data = json.load(f)

    print(len(old_data))

    result = []
    err_cnt = 0

    for i in tqdm(range(len(old_data))):
        question = old_data[i]['question']
        references_docs = old_data[i]['docs'][:args.docs]
        references_sentences = []
        reference_text = ""
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

        if args.shot == 3:
            prompt = generate_direct_prompt_shot3(question, reference_text)
        else:
            prompt = generate_direct_prompt_shot2(question, reference_text)
        
        output = llama3_generate(prompt)
        print(output)

        old_data[i].update({
            "question": question,
            "references": references_docs,
            "gold_answer": gold_answer,
            "pred_answer": output.strip()
        })

        result.append(old_data[i])

    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=4)