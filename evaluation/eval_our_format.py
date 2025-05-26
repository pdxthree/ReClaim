from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)
import argparse
import collections
import json
import re
import string
import torch
import copy
import pandas as pd
from nltk import sent_tokenize
import numpy as np
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm
import sys
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


QA_MODEL = "gaotianyu1350/roberta-large-squad"

AUTOAIS_MODEL="t5_xxl_true_nli_mixture"
device = "cuda:0"
device_id = 0

global autoais_model, autoais_tokenizer
autoais_model, autoais_tokenizer = None, None

def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def compute_f1(a_gold, a_pred):
    """Compute F1 score between two strings."""

    def _get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()

    gold_toks = _get_tokens(a_gold)
    pred_toks = _get_tokens(a_pred)

    print(gold_toks, pred_toks)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_exact(a_gold, a_pred):
    """Check whether two strings are equal up to normalization."""

    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """

    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False


def compute_rouge(data):
    """Main function for rouge scoring.
    If two references are provided,
    the best score is chosen for each instance.
    Args:
        data: requires field `output` and `answer` (or `annotations` for ASQA)
        metrics: list of evaluation metrics
    Returns:
        dictionary representation of rouge scores
    """
    def _rouge_calculation(hypotheses,
                           references1,
                           references2=[],
                           metrics=['rougeLsum']):

        if references2 == []:
            references2 = references1

        scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()

        for i in range(len(hypotheses)):
            scores1 = scorer.score(references1[i], hypotheses[i])
            scores2 = scorer.score(references2[i], hypotheses[i])
            if scores1['rougeLsum'].fmeasure > scores2['rougeLsum'].fmeasure:
                aggregator.add_scores(scores1)
            else:
                aggregator.add_scores(scores2)

        scores = {m: [] for m in metrics}

        for m in metrics:
            fmeasure = aggregator.aggregate()[m].mid.fmeasure
            scores[m].append(fmeasure)

        for m in scores:
            scores[m] = 100 * sum(scores[m]) / len(scores[m])

        return scores

    hypotheses = {}
    references1 = {}
    references2 = {}

    for idx, item in enumerate(data):
        hypotheses[idx] = item["output"]
        if "annotations" in item and item['annotations'] is not None and item['annotations'] != []:  # For ASQA
            references1[idx] = item["annotations"][0]["long_answer"]
            references2[idx] = item["annotations"][1]["long_answer"]
        else:
            references1[idx] = item["answer"]
            references2[idx] = item["answer"]

    h, r1, r2 = [], [], []

    for key in references1:
        h.append(hypotheses[key])
        r1.append(references1[key])

        if references2 is not None:
            r2.append(references2[key])

    h = ['\n'.join(sent_tokenize(text.lower())) for text in h]
    r1 = ['\n'.join(sent_tokenize(text.lower())) for text in r1]
    r2 = ['\n'.join(sent_tokenize(text.lower())) for text in r2]
    scores = _rouge_calculation(h, r1, r2)

    return scores['rougeLsum']


def compute_str_em(data):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """

    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        return 0, 0

    acc = []
    hit = []

    for item in data:
        loc_acc = []
        for qa_pair in item['qa_pairs']:
            loc_acc.append(exact_presence(
                qa_pair['short_answers'], item["output"]))
        acc.append(np.mean(loc_acc))
        hit.append(int(np.mean(loc_acc) == 1))

    return 100 * np.mean(acc), 100 * np.mean(hit)


def compute_len(data):
    """Compute average length of predictions."""

    res, cntr = 0, 0
    for item in data:
        res += len(item["output"].split())
        cntr += 1
    return res / cntr


def compute_qa(data):
    """Compute QA-based accuracy.
    Args:
        data: requires filed `qa_pairs/short_answers` and `output`
    Returns:
        QA metrics (QA-EM, QA-F1, QA-Hit)
    """

    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        logger.warn("Warning: no QA pairs found in data")
        return {
            'QA-EM': 0,
            'QA-F1': 0,
            'QA-Hit': 0,
        }

    logger.info("Loading the RoBERTa-large SQuAD model for QA-based accuracy...")
    qa_pipeline = pipeline("question-answering", model=QA_MODEL, device=device)
    logger.info("Done")

    logger.info("Computing the QA-based accuracy...")
    em, f1, bins = [], [], []
    for item in tqdm(data):
        question = [qa_pair['question'] for qa_pair in item['qa_pairs']]
        context = item['output'] if len(item['output']) > 0 else " "
        results = qa_pipeline(
            question=question, context=context, handle_impossible_answer=True)
        loc_counter, loc_em, loc_f1 = 0, 0, 0

        for idx, res in enumerate(results):
            answers = item["qa_pairs"][idx]["short_answers"]
            prediction = res["answer"]

            loc_em += max([compute_exact(a, prediction) for a in answers])
            loc_f1 += max([compute_f1(a, prediction) for a in answers])
            loc_counter += 1

        em.append(loc_em / loc_counter)
        f1.append(loc_f1 / loc_counter)
        bins.append(loc_em == loc_counter)

    torch.cuda.empty_cache()
    qa_pipeline = None
    torch.cuda.empty_cache()

    return {
        'QA-EM': 100 * np.mean(em),
        'QA-F1': 100 * np.mean(f1),
        'QA-Hit': 100 * np.mean(bins)
    }


def compute_mauve(data):
    """Compute Mauve score."""

    logger.info("Computing MAUVE...")
    human_data = []
    model_data = []
    for item in data:
        human_data.append(' '.join(
            (item['question'] + " " + item['answer'].strip()).split()[:100]).rstrip(string.punctuation))
        model_data.append(' '.join(
            (item['question'] + " " + item['output'].strip()).split()[:100]).rstrip(string.punctuation))

    import mauve
    out = mauve.compute_mauve(
        p_text=human_data,
        q_text=model_data,
        device_id=device_id,
        max_text_length=512,
        verbose=True,
        batch_size=8,
        featurize_model_name="gpt2-large"
    )

    torch.cuda.empty_cache()

    return out.mauve * 100


def _run_nli_autoais(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    global autoais_model, autoais_tokenizer
    input_text = "premise: {} hypothesis: {}".format(passage, claim)
    input_ids = autoais_tokenizer(
        input_text, return_tensors="pt").input_ids.to(autoais_model.device)
    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    return inference


def compute_claims(data):
    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
            AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map=device)
        autoais_tokenizer = AutoTokenizer.from_pretrained(
            AUTOAIS_MODEL, use_fast=False)

    logger.info("Computing claims...")
    scores = []
    for item in tqdm(data):
        normalized_output = remove_citations(item['output'])
        entail = 0
        claims = item["claims"]
        for claim in claims:
            entail += _run_nli_autoais(normalized_output, claim)
        scores.append(entail / len(claims))

    torch.cuda.empty_cache()

    return 100 * np.mean(scores)


def compute_autoais(data,
                    decontext=False,
                    concat=False,
                    qampari=False,
                    at_most_citations=None,):
    """
    Compute AutoAIS score.

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
        decontext: decontextualize the output
    """

    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
            AUTOAIS_MODEL, torch_dtype=torch.bfloat16,  device_map=device)
        autoais_tokenizer = AutoTokenizer.from_pretrained(
            AUTOAIS_MODEL, use_fast=False)

    logger.info(f"Running AutoAIS...")

    ais_scores = []
    ais_scores_prec = []
    new_file = []
    sent_total = 0
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0

    cnt = 0
    for item in tqdm(data):
        references = item['references']
        reference_list = []
        for reference in references:
            reference_list.append([x.strip()
                                  for x in sent_tokenize(reference['text'])])
        nli_result = []
        cnt = 0
        total_citations = 0
        entail_prec = 0
        
        if len(item['used_references']) == 0 or len(item['used_references']) != len(item['pred_sentences']):
            cnt += 1
            continue
        for i in range(len(item['used_references'])):
            used_reference = item['used_references'][i].strip()
            pred_sentence = item['pred_sentences'][i].strip()
            temp = _run_nli_autoais(used_reference, pred_sentence)
            nli_result.append(temp)
            if temp == 1:
                cnt += 1

            
            citation_sentences = [x.strip()
                                  for x in sent_tokenize(used_reference)]

            total_citations += len(citation_sentences)
            print(len(citation_sentences), temp, citation_sentences)
            if temp == 1 and len(citation_sentences) > 1:
                sent_mcite_support += 1
                for citation_sentence in citation_sentences:
                    nli_tmp = _run_nli_autoais(
                        citation_sentence, pred_sentence)

                    if not nli_tmp:
                        subset_exclude = copy.deepcopy(citation_sentences)
                        passage = '\n'.join(
                            [x for x in subset_exclude if x != citation_sentence])
                        nli_subset = _run_nli_autoais(passage, pred_sentence)
                        if nli_subset: 
                            sent_mcite_overcite += 1
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += temp
            print(entail_prec, total_citations)
        print(cnt/len(item['used_references']))
        sent_total += len(item['used_references'])
        item['nli_result'] = nli_result
        item['nli_score'] = cnt/len(item['used_references'])
        item['nli_pre'] = entail_prec / \
            total_citations if total_citations > 0 else 0
        ais_scores.append(cnt/len(item['used_references']))
        ais_scores_prec.append(
            entail_prec / total_citations if total_citations > 0 else 0) 
        new_file.append(item)
    print(cnt)

    data = copy.deepcopy(new_file)

    if sent_mcite > 0 and sent_mcite_support > 0:
        print("Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite." % (
            100 * sent_mcite / sent_total,
            100 * sent_mcite_support / sent_mcite,
            100 * sent_mcite_overcite / sent_mcite_support
        ))

    return {
        "citation_rec": 100 * np.mean(ais_scores),
        "citation_prec": 100 * np.mean(ais_scores_prec),
    }


def compute_qampari_f1(data, cot=False):
    prec = []
    rec = []
    rec_top5 = []
    f1 = []
    f1_top5 = []

    num_preds = []
    for item in data:
        if cot:
            if ":" in item['output']:
                o = ':'.join(item['output'].split(":")[1:])
            else:
                o = ""
        else:
            o = item['output']
        preds = [normalize_answer(x.strip()) for x in o.rstrip().rstrip(
            ".").rstrip(",").split(",")]
        preds = [p for p in preds if len(p) > 0]
        num_preds.append(len(preds))
        answers = [[normalize_answer(x) for x in ans]
                   for ans in item['answers']]
        flat_answers = [item for sublist in answers for item in sublist]

        prec.append(sum([p in flat_answers for p in preds]) /
                    len(preds) if len(preds) > 0 else 0)
        rec.append(sum([any([x in preds for x in a])
                   for a in answers]) / len(answers))
        rec_top5.append(min(5, sum([any([x in preds for x in a])
                        for a in answers])) / min(5, len(answers)))
        if (prec[-1] + rec[-1]) == 0:
            f1.append(0)
        else:
            f1.append(2 * prec[-1] * rec[-1] / (prec[-1] + rec[-1]))
        if (prec[-1] + rec_top5[-1]) == 0:
            f1_top5.append(0)
        else:
            f1_top5.append(2 * prec[-1] * rec_top5[-1] /
                           (prec[-1] + rec_top5[-1]))

    return {
        "num_preds": np.mean(num_preds),
        "qampari_prec": 100 * np.mean(prec),
        "qampari_rec": 100 * np.mean(rec),
        "qampari_rec_top5": 100 * np.mean(rec_top5),
        "qampari_f1": 100 * np.mean(f1),
        "qampari_f1_top5": 100 * np.mean(f1_top5),
    }


def citation_proportion(data):
    used_len = 0
    sum_len = 0
    reference_cnt = 0
    multi_reference = 0
    verbatim_citation = 0

    for item in data:
        reference = ""
        for tmp in item['references']:
            reference = reference + tmp['text'] + " "
        reference = reference.strip()
        sum_len += len(reference)
        used_references = item['used_references']
        cnt = 0
        reference_cnt += len(used_references)
        for used_reference in used_references:
            cnt += len(used_reference)
            
            references_list = [x.strip() for x in re.split(
                r'(\. |\! |\? |\; |[。？！；])', used_reference)]
            references_list = [references_list[p] + references_list[p + 1]
                               for p in range(0, len(references_list) - 1, 2)]
            flag = 0
            for reference_item in references_list:
                if reference_item.strip() in reference:
                    continue
                else:
                    flag = 1
            if flag == 0:
                verbatim_citation += 1
            if used_reference.strip() not in reference:
                multi_reference += 1
        print(cnt)
        used_len += cnt

    return verbatim_citation/reference_cnt, used_len/sum_len, multi_reference/reference_cnt


def irrelevant_reference(data):
    total_reference = 0
    irrelevant_reference = 0
    irrelevant_source = 0
    for item in data:
        Irrelevant_index = []
        for i in range(len(item['references'])):
            doc = item['references'][i]
            print(doc)
            if ("extraction" in doc and ("Irrelevant" in doc['extraction'] or "irrelevant" in doc['extraction'])) or ("summary" in doc and ("Irrelevant" in doc['summary'] or "irrelevant" in doc['summary'])):
                Irrelevant_index.append(i)
        total_reference += len(item['used_references'])
        irrelevant_source += len(Irrelevant_index)
        for used_reference in item['used_references']:
            
            references_list = [x.strip() for x in re.split(
                r'(\. |\! |\? |\; |[。？！；])', used_reference)]
            references_list = [references_list[p] + references_list[p + 1]
                               for p in range(0, len(references_list) - 1, 2)]
            irr_set = set()
            for reference_item in references_list:
                for idx in range(len(item['references'])):
                    if reference_item in item['references'][idx]['text'] and idx in Irrelevant_index:
                        irr_set.add(idx)
            irrelevant_reference += len(irr_set)

    print(irrelevant_reference)
    print(total_reference)
    print(irrelevant_reference/total_reference)
    print(irrelevant_reference/irrelevant_source)

    return irrelevant_reference/total_reference, irrelevant_reference/irrelevant_source


def deal_output(data):
    new_data = []
    res, cntr = 0, 0
    for item in data:
        item['used_references'] = [x.strip() for x in re.findall(
            r'<reference>(.*?)</reference>', item['pred_answer'])]
        used_references_len = 0
        for used_reference in item['used_references']:
            if used_reference != "":
                used_references_len += 1
            else:
                break
        item['pred_sentences'] = [x.strip() for x in re.findall(
            r'<claim>(.*?)</claim>', item['pred_answer'])]
        pred_sentences_len = len(item['pred_sentences'])
        useful_len = 0
        if used_references_len < pred_sentences_len:
            useful_len = used_references_len
        else:
            useful_len = pred_sentences_len
        output = ""
        for tmp in item['pred_sentences'][:useful_len]:
            output = output + tmp + " "
        item['output'] = output.strip()
        item['used_references'] = item['used_references'][:useful_len]
        for ref in item['used_references']:
            res += len(ref.split())

        item['pred_sentences'] = item['pred_sentences'][:useful_len]
        new_data.append(item)
    return new_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, required=True, help="Output file. Should have field `question`, `output`, (ROUGE) `answer`, \
                        (accuracy) `qa_pairs`, (AIS) `docs`")
    parser.add_argument("--no_rouge", action="store_true",
                        help="Do not evaluate ROUGE score")
    parser.add_argument("--qa", action="store_true", help="Use the QA model")
    parser.add_argument("--mauve", action="store_true",
                        help="Use the mauve score model")
    parser.add_argument("--citations", action="store_true",
                        help="Evaluation with citation")
    parser.add_argument("--at_most_citations", type=int, default=5,
                        help="At most take this many documents (mostly for precision)")
    parser.add_argument("--claims_nli", action="store_true",
                        help="Use claims for ELI5")

    parser.add_argument("--cot", action="store_true",
                        help="For QAMPARI, try to find colon and separate the COT and answer listing")

    args = parser.parse_args()

    with open(args.f) as f:
        data_with_config = json.load(f)
    data = data_with_config

    
    data = deal_output(data)

    if "qampari" in args.f:
        args.no_rouge = True
        args.qa = False
        args.mauve = False
        args.decontext = False
        qampari = True
    else:
        qampari = False

    logger.warning(
        "We remove all the pre/appended space/newlines and we truncate the answer by the first newline.")
    logger.warning(
        "We replace any on the fly search result to standard bracket citation format.")
    for i in range(len(data)):
        data[i]['output'] = data[i]['output'].strip().split("\n")[0]
        data[i]['output'] = data[i]['output'].replace("<|im_end|>", "")
        data[i]['references'] = data[i]['docs']

    normalized_data = copy.deepcopy(data)
    for i in range(len(normalized_data)):
        normalized_data[i]['output'] = remove_citations(
            normalized_data[i]['output'])

    result = {}
    result['length'] = compute_len(normalized_data)
    result['str_em'], result['str_hit'] = compute_str_em(normalized_data)

    if qampari:
        result.update(compute_qampari_f1(normalized_data, cot=args.cot))
    if not args.no_rouge:
        result['rougeLsum'] = compute_rouge(normalized_data)
    if args.qa:
        result.update(compute_qa(normalized_data))
    if args.mauve:
        result['mauve'] = compute_mauve(normalized_data)
    if args.citations:
        result.update(compute_autoais(data, qampari=qampari,
                      at_most_citations=args.at_most_citations))
    if args.claims_nli:
        result["claims_nli"] = compute_claims(normalized_data)

    print(result)
    json.dump(result, open(args.f + "_our" + ".score", "w"), indent=4)


if __name__ == "__main__":
    main()