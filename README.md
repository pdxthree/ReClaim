<div align= "center">
    <h1> ReClaim </h1>
</div>

<p align="center">  
Ground Every Sentence: Improving Retrieval-Augmented LLMs with Interleaved Reference-Claim Generation
</p>

<p align="center">  
<a href="https://arxiv.org/pdf/2407.01796">Paper</a>; 
</p>

Here is the open-source code repository for the paper "Ground Every Sentence: Improving Retrieval-Augmented LLMs with Interleaved Reference-Claim Generation."

<p align="center">
  <img src="https://github.com/user-attachments/assets/5174d348-9454-4500-9fef-42c656af8425" alt="描述文本" width="300">
</p>

## Abstract
Retrieval-Augmented Generation (RAG) has been widely adopted to enhance Large Language Models (LLMs) in knowledge-intensive tasks. To enhance credibility and verificability in RAG systems, Attributed Text Generation (ATG) is proposed, which provides citations to retrieval knowledge in LLM-generated responses. Prior methods mainly adopt coarse-grained attributions, with passage-level or paragraph-level references or citations, which fall short in verificability. This paper proposes ReClaim(Refer & Claim), a fine-grained ATG method that alternates the generation of references and answers step by step. Different from previous coarse-grained attribution, ReClaim provides sentence-level citations in long-form question-answering tasks. With extensive experiments, we verify the effectiveness of ReClaim in extensive settings, achieving a citation accuracy rate of 90%.

## Method
![image](https://github.com/user-attachments/assets/ebee1835-dca3-4dd2-8ba6-8bfd540ed825)

## Experiment
![image](https://github.com/user-attachments/assets/88a8216f-9c26-4f54-9636-8d20a095852d)
![image](https://github.com/user-attachments/assets/7f4d35ac-3757-4021-973e-129defd7b13f)
![image](https://github.com/user-attachments/assets/ffd7b9de-053b-41ce-b010-57e2443c3081) ![image](https://github.com/user-attachments/assets/9f71326b-0a3a-42dc-b839-e61733b18692)

## Citation

Feel free to cite us if you like ReClaim.
```bibtex
@article{xia2024ground,
  title={Ground Every Sentence: Improving Retrieval-Augmented LLMs with Interleaved Reference-Claim Generation},
  author={Xia, Sirui and Wang, Xintao and Liang, Jiaqing and Zhang, Yifei and Zhou, Weikang and Deng, Jiaji and Yu, Fei and Xiao, Yanghua},
  journal={arXiv preprint arXiv:2407.01796},
  year={2024}
}
```
