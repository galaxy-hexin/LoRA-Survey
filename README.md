# Awesome-LLM-LoRA
[![](https://img.shields.io/badge/📑-Survey_Paper-blue)](https://arxiv.org/abs/2404.03354)
[![Awesome](https://awesome.re/badge.svg)](https://github.com/TUDB-Labs/Awesome-LLM-LoRA)
![](https://img.shields.io/github/last-commit/TUDB-Labs/Awesome-LLM-LoRA?color=green) 
![](https://img.shields.io/badge/PRs-Welcome-red)
![](https://img.shields.io/github/stars/TUDB-Labs/Awesome-LLM-LoRA?color=yellow)
![](https://img.shields.io/github/forks/TUDB-Labs/Awesome-LLM-LoRA?color=lightblue)

A collection of papers and resources related to LoRA and its variants

Parameter-efficient fine-tuning (PEFT) aims to accomplish the training of large models on downstream tasks by adjusting only a small subset of parameters. LoRA achieves this by applying low-rank approximations to the weights, vastly reducing the number of parameters required for fine-tuning while still approaching the performance levels of full fine-tuning.

## Table Content
- [TableContent](README.md#table-content)
- [Timeline of LoRA](README.md#timeline-of-lora)
- [Overview of the LoRA Family](README.md#overview-of-the-lora-family)
- [Experiments](README.md#experiments)
- [Paper List](README.md#paper-list)
    - [LoRA Related](README.md#lora-related)
    - [MoE LoRA](README.md#moe-lora)
- [Acknowledge](README.md#acknowledge)
- [Update Log](README.md#update-log)
- [reference](README.md#reference)

## Timeline of LoRA

## Overview of the LoRA Family
|LoRA           |     Characterization                  |  |    |
|---------------|--------------------------------------|--|----|
|LoRA+          |Efficiency Improvement: LoRA+ is an effective low-rank adaptation method that adjusts LoRA adapter matrices A and B with different learning rates, enhancing the efficiency of feature learning. It demonstrates superior performance (1%-2% improvement) and faster fine-tuning speed (up to ~2x speedup) compared to LoRA in large models, while maintaining similar computational costs.|Learning Rate Setting: The key innovation of LoRA+ lies in assigning distinct learning rates to matrices A and B of the LoRA adapter, correcting suboptimality by choosing an appropriate fixed ratio. In LoRA+, these matrices have their learning rates set differently to boost feature learning efficiency.|Experimental Validation: Extensive experimental studies show significant enhancements in both performance and fine-tuning speed with LoRA+, all while keeping computational costs on par with LoRA. The improvements facilitate more efficient feature learning during the fine-tuning of large models, thereby augmenting model performance and speed.
|PeriodicLoRA   |Parameter Efficiency: PeriodicLoRA (PLoRA) is a parameter-efficient fine-tuning method that accumulates low-rank update matrices multiple times for higher update ranks without increasing memory usage.|Cyclical Unloading Strategy: PLoRA employs a momentum-based unloading strategy, periodically offloading the results of LoRA weight training into the backbone parameters and then reinitializing the LoRA state, including weights, optimizer states, and learning rate schedulers.|Multi-stage Training: PLoRA undergoes several training stages, unloading LoRA weights to backbone parameters and resetting LoRA state at each stage's conclusion. This approach allows PLoRA to transcend the limitations of low-rank update matrices, approximating full fine-tuning efficacy.
|VeRA           |Vector-based Random Matrix Adaptation (VeRA): This proposed method, VeRA, significantly reduces the number of trainable parameters by employing a pair of shared low-rank matrices and learning small scaling vectors, maintaining performance levels akin to LoRA. VeRA validates its effectiveness across GLUE and E2E benchmarks, image classification tasks, and instruction fine-tuning on 7B and 13B language models.|Low Storage Challenge: The paper highlights that traditional Low-rank adaptation (LoRA) still encounters storage challenges when fine-tuning large language models, especially when scaling to larger models or deploying for numerous users or task adaptations. VeRA addresses this challenge by significantly reducing trainable parameters through shared low-rank matrices and small scaling vectors.|No Additional Inference Time Cost: VeRA, a novel fine-tuning method, incurs no extra inference time cost. Compared to LoRA, VeRA further decreases the number of trainable parameters while achieving comparable outcomes. Its performance is benchmarked against LoRA and other parameter-efficient adaptation methods on natural language understanding (GLUE) and generation (E2E), and in instruction following and image classification tasks.
|LoRA-FA        |LoRA-FA: LoRA-FA is a memory-efficient fine-tuning method designed to reduce activation memory usage without compromising performance or necessitating expensive recomputation. It freezes the projection-down weights A in each LoRA layer and updates the projection-up weights B. This ensures changes to model weights remain within a low-rank space during LLM fine-tuning, eliminating the need to store full-rank input activations. LoRA-FA has been extensively tested across various model types (RoBERTa, T5, LLaMA) and scales.|Experimental Outcomes: Results show that LoRA-FA consistently achieves comparable fine-tuning accuracy across tasks, while reducing overall memory costs by up to 1.4x compared to full parameter fine-tuning and LoRA.|Memory Efficiency: LoRA-FA targets the memory challenges of fine-tuning large language models by freezing projection-down weights A and updating projection-up weights B, effectively reducing activation memory usage while preserving good fine-tuning performance. The design of LoRA-FA ensures model weight changes occur within a low-rank space, reducing memory consumption and enhancing efficiency during fine-tuning.|Performance Preservation: A key feature of LoRA-FA is maintaining performance while reducing memory costs. By selectively freezing and updating specific projection weights, LoRA-FA ensures stability and accuracy during the fine-tuning process, while reducing the demand for costly activation memory. Experiments demonstrate its accuracy comparable to full parameter fine-tuning and LoRA, with significantly reduced memory costs.
|LoRA-drop      |LoRA-drop: LoRA-drop is an effective pruning method for LoRA parameters, assessing parameter importance via LoRA outputs. It retains LoRA for important layers while sharing parameters among LoRA for other layers. Extensive experiments on natural language understanding (NLU) and natural language generation (NLG) tasks validate its effectiveness.|Parameter Efficiency: LoRA-drop aims to enhance LoRA's parameter efficiency further by evaluating and pruning parameters based on LoRA output analysis. The design of LoRA-drop accounts for the direct impact of LoRA output-related parameters on the frozen model, enabling effective pruning and higher parameter utilization efficiency.|Experimental Validation: LoRA-drop, through extensive experimentation on multiple NLU and NLG tasks, demonstrates results on par with the original LoRA method while using only 50% of LoRA parameters. Further analysis confirms the effectiveness of LoRA-drop.
|AdaLoRA        |AdaLoRA: AdaLoRA is a method of adaptive parameter allocation for parameter-efficient fine-tuning. It allocates parameter budgets based on importance scores of weight matrices. By parametrizing incremental updates in singular value decomposition (SVD) form, it efficiently prunes insignificant singular values, reducing their parameter budget without costly exact SVD computations. AdaLoRA's effectiveness is widely validated across natural language processing (NLP), question answering, and NLG domains.|Parameter Efficiency: AdaLoRA aims to improve parameter efficiency by adaptively allocating parameter budgets according to importance ratings of weight matrices. It effectively trims unimportant updates, lowering their parameter budget while enhancing fine-tuning performance. AdaLoRA's design focuses on the importance of weight parameters for more efficient allocation.|Experimental Validation: AdaLoRA's wide-ranging experiments across NLP, question answering, and NLG domains verify its efficacy. It significantly outperforms baseline methods under low budget settings, demonstrating substantial improvements across multiple tasks and pre-trained models.
|DoRA           |Weight-Decomposed Low-Rank Adaptation (DoRA): DoRA is a low-rank adaptation method that decomposes pre-training weights into magnitude and direction components for fine-tuning. It uses LoRA for directional updates, effectively reducing the number of trainable parameters while enhancing LoRA's learning capacity and training stability. It excels in diverse downstream tasks such as common-sense reasoning, visual instruction tuning, and image/video-text understanding.|Enhanced Learning Capacity: DoRA seeks to augment LoRA's learning capacity by exploring the intrinsic differences between full parameter fine-tuning (FT) and LoRA through weight decomposition. By separately fine-tuning magnitude and direction components, it strengthens LoRA's learning capacity without additional inference overhead.|Training Stability: DoRA improves LoRA's training stability through weight decomposition and directional updates, avoiding the expensive training costs associated with full parameter fine-tuning. Experimental differences highlight its effectiveness in parameter-efficient fine-tuning.
|Delta-LoRA     |Delta-LoRA: Delta-LoRA is a novel parameter-efficient fine-tuning method for large language models (LLMs). Unlike LoRA and other low-rank adaptation methods like AdaLoRA, Delta-LoRA not only updates low-rank matrices A and B but also propagates learning to pre-training weights W through incremental updates of the product of two low-rank matrices (A(t+1)B(t+1) − A(t)B(t)). This strategy addresses the limitation of low-rank matrices' incremental updates in capturing representations suitable for downstream tasks. Moreover, since updating W does not require computing gradients or storing their momenta, Delta-LoRA maintains comparable memory requirements and computational costs with LoRA. Extensive experiments demonstrate Delta-LoRA's superiority over existing low-rank adaptation methods.|Enhanced Learning Capacity: Delta-LoRA aims to enhance LoRA's learning capability through weight decomposition and incremental updates. By decomposing weights into magnitude and direction components and fine-tuning both, it bolsters LoRA's learning power, showcasing superior performance in tasks like common-sense reasoning, visual instruction tuning, and multimedia text understanding.|Memory and Computational Efficiency: Delta-LoRA maintains memory and computational efficiency by effectively updating pre-training weights W without requiring gradient computations or momentum storage. This approach, while boosting learning capability, sustains the efficiency needed for fine-tuning large language models.|

## Experiments
We explored the fine-tuning effects of various versions of LoRA and MoE LoRA on different models across various datasets. We endeavored to ensure that different variants of LoRA had similar numbers of parameters when fine-tuning the same model for ease of comparison. \
For details of the experimental process, please click <u>[here](./experiment/README.md)</u>

Here are the results of our experiments:

![experiment](assets/experiments.png)

## Paper List
### LoRA Related
- (arXiv'2021) LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS [ [paper](https://arxiv.org/pdf/2106.09685) ] [ [repo](https://github.com/microsoft/LoRA) ]
- (arXiv'2024) LoRA+: Efficient Low Rank Adaptation of Large Models [ [paper](https://arxiv.org/pdf/2402.12354.pdf) ] [ [repo](https://github.com/nikhil-ghosh-berkeley/loraplus) ]
- (arXiv'2024) PeriodicLoRA: Breaking the Low-Rank Bottleneck in LoRA Optimization [ [paper](https://arxiv.org/pdf/2402.16141.pdf) ]
- (arXiv'2023) VeRA：VERA: VECTOR-BASED RANDOM MATRIX ADAPTATION [ [paper](https://arxiv.org/pdf/2310.11454.pdf) ]
- (arXiv'2023) LORA-FA: MEMORY-EFFICIENT LOW-RANK ADAPTATION FOR LARGE LANGUAGE MODELS FINE-TUNING [ [paper](https://arxiv.org/pdf/2308.03303.pdf) ]
- (arXiv'2024) LoRA-drop: Efficient LoRA Parameter Pruning based on Output
Evaluation [ [paper](https://arxiv.org/pdf/2402.07721.pdf) ]
- (arXiv'2023)ADALORA: ADAPTIVE BUDGET ALLOCATION FOR
PARAMETER-EFFICIENT FINE-TUNING [ [paper](https://arxiv.org/pdf/2303.10512.pdf) ] [ [repo](https://github.com/qingruzhang/adalora) ]
- (arXiv'2024) DoRA: Weight-Decomposed Low-Rank Adaptation [ [paper](https://arxiv.org/pdf/2402.09353.pdf) ] [ [repo](https://github.com/nbasyl/DoRA) ]
- (arXiv'2023) DELTA-LORA: FINE-TUNING HIGH-RANK PARAMETERS WITH THE DELTA OF LOW-RANK MATRICES[ [paper](https://arxiv.org/pdf/2309.02411.pdf) ]
- DyLoRA [ [paper](https://arxiv.org/pdf/2210.07558) ]
- IncreLoRA: 2308.12043 (arxiv.org)
- LoRAPrune: 2305.18403 (arxiv.org)
- QA-LORA:2309.14717 (arxiv.org)
LoftQ: 2310.08659 (arxiv.org)
Bayesian Low-rank Adaptation: 2308.13111 (arxiv.org)
LoraHub: 2307.13269 (arxiv.org)
LQ-LoRA: 2311.12023 (arxiv.org)
RST-LoRA: 2405.00657 (arxiv.org)
HydraLoRA: 2404.19245 (arxiv.org)
FeDeRA: 2404.18848 (arxiv.org)
FLoRA: 2404.15182 (arxiv.org)


### MoE LoRA
- (arXiv'2024) MoELoRA: Contrastive Learning Guided Mixture of Experts on Parameter-Efficient Fine-Tuning for Large Language Models [ [paper](https://arxiv.org/pdf/2402.12851.pdf) ] [ [repo]() ]
- (arXiv'2024) PESC: Parameter-Efficient Sparsity Crafting rom Dense to Mixture-of-Experts for Instruction Tuning on General Tasks [ [paper](https://arxiv.org/pdf/2401.02731.pdf) ] [ [repo](https://github.com/wuhy68/Parameter-Efficient-MoE) ]
- (arXiv'2024) Higher Layers Need More LoRA Experts [ [paper](https://arxiv.org/pdf/2402.08562.pdf) ] [ [repo](https://github.com/GCYZSL/MoLA) ]
- LORAMOE: REVOLUTIONIZING MIXTURE OF EXPERTS FOR MAINTAINING WORLD KNOWLEDGE IN LANGUAGE MODEL ALIGNMENT [ [paper](https://simg.baai.ac.cn/paperfile/96f0cfd7-79c7-4110-88e5-4ea80a7fbc8d.pdf) ] [ [repo](https://github.com/Ablustrund/LoRAMoE) ]
- (arXiv'2023) MOELoRA: An MOE-based Parameter Efficient Fine-Tuning Method for Multi-task Medical Applications [ [paper](https://arxiv.org/pdf/2310.18339.pdf) ] [ [repo](https://github.com/liuqidong07/MOELoRA-peft) ]
- (arXiv'2024) Mixture-of-LoRAs: An Efficient Multitask Tuning for Large Language Models [ [paper](https://arxiv.org/pdf/2403.03432.pdf) ] [ [repo]() ]
- MoSA: Mixture of Sparse Adapters for Visual Efficient Tuning [ [paper](https://www.researchgate.net/profile/Bocheng_Zou/publication/376312824_MoSA_Mixture_of_Sparse_Adapters_for_Visual_Efficient_Tuning/links/660b760e390c214cfd2fa277/MoSA-Mixture-of-Sparse-Adapters-for-Visual-Efficient-Tuning.pdf) ]
- (arXiv'2024) MoRAL: MoE Augmented LoRA for LLMs’ Lifelong Learning [ [paper](https://arxiv.org/abs/2402.11260) ] [ [repo]() ]
- (arXiv'2023) Mixture of Cluster-conditional LoRA Experts for Vision-language Instruction Tuning [ [paper](https://arxiv.org/pdf/2312.12379.pdf) ] [ [repo](https://gyhdog99.github.io/projects/mocle/) ]
- (arXiv'2024) Mixture of Experts for Large Vision-Language Models [ [paper]](https://arxiv.org/pdf/2401.15947.pdf) ] [ [repo](https://github.com/PKU-YuanGroup/MoE-LLaVA) ]
- MIXTURE OF LORA EXPERTS [ [paper](https://openreview.net/pdf?id=uWvKBCYh4S) ] [ [repo]() ]
- (arXiv'2024) Multi-Task Dense Prediction via Mixture of Low-Rank Experts [ [paer](https://arxiv.org/pdf/2403.17749.pdf) ] [ [repo](https://github.com/YuqiYang213/MLoRE) ]
- (arXiv'2023) OCTAVIUS: MITIGATING TASK INTERFERENCE IN MLLMS VIA LORA-MOE [ [paper](https://arxiv.org/pdf/2311.02684.pdf) ] [ [repo]() ]
- (arXiv'2024) Multimodal Instruction Tuning with Conditional Mixture of LoRA [ [paper](https://arxiv.org/pdf/2402.15896.pdf) ] [ [repo]() ]
- (arXiv'2022) AdaMix: Mixture-of-Adaptations for Parameter-efficient Model Tuning [ [paper](https://arxiv.org/pdf/2205.12410.pdf) ] [ [repo]() ]
- Combining Parameter-efficient Modules for Task-level Generalisation [ [paper](https://aclanthology.org/2023.eacl-main.49.pdf) ] [ [repo]() ]
- (arXiv'2023) Sparse Mixture of Low Rank Adaptation [ [paper](https://arxiv.org/pdf/2311.09179.pdf) ] [ [repo]() ]
- (arXiv'2024)LLaVA-MoLE: Sparse Mixture of LoRA Experts for Mitigating Data Conflicts in Instruction Finetuning MLLMs [ [paper](https://arxiv.org/pdf/2401.16160.pdf) ] [ [repo]() ]

## Acknowledge

## Update Log
| Version                  | Time       | Update Content                                               |
| ------------------------ | ---------- | ------------------------------------------------------------ |
| V1                       | 2024/4/20 | The initial version.                                         |

## reference
