# Skill-LM
This is the repository hosting the pre-trained models, training and evaluation codebase, and datasets for the ACL 2023 main conference paper 'Learning Non-linguistic Skills without Sacrificing Linguistic Proficiency'

> **Note:** As the pre-trained PyTorch models are larger in size, [Git LFS](https://git-lfs.github.com/) was used to push these models to Git. Please make sure you have it installed in your system.

Before using this repository, please make sure you have the following packages installed in your environment:
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Tqdm](https://github.com/tqdm/tqdm)
- [Sklearn](https://scikit-learn.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Jupyter Notebook](https://jupyter.org/)
- [PyTorch](https://pytorch.org/)
- [Huggingface Transformers & Dataset](https://huggingface.co/)

This environment can be built simply using the requirements file supplied in this repository. Inside your virtual environment, please run:


```
pip install -r requirements.txt
```

> **Note:** The versions of PyTorch (and subsequently Huggingface) required for GPU-use is highly dependent on the user GPU and CUDA version.

Please follow the step-wise description below to replicate the results published in 'Learning Non-linguistic Skills without Sacrificing Linguistic Proficiency'.

## Repository Structure
Ready-to-use pre-trained versions of all 4 models used in this paper (including the baseline) are included inside the 'Models' directory:
- BERT: The base BERT model trained for quantitative reasoning conventionally using cross-entropy loss. Referred to as BERT<sub>Arith</sub> in the paper.
- BERT_Regression: The base BERT model trained for quantitative reasoning using a combination of regression loss and cross-entropy loss. Referred to as Skill-LM<sub>(w/o Lewc)</sub> in the paper.
- Skill-LM: Our skill-empowered BERT model trained for quantitative and linguistic reasoning. 

The 'Dataset' directory consists of 2 sub-directories:
- Arithmetic: This is a randomly sampled subset (n/4) of [GenBERT's](https://aclanthology.org/2020.acl-main.89/) training data.
- Generalization: This directory contains out-of-domain samples that are greater than the [0, 20000] range of the original training data.

The 'Fisher' directory contains scripts for the fisher matrix computations for model variants.

## Loading and Evaluating the Pre-trained LLMs on Quantitative Reasoning
Without any GPU-based training, you can replicate the results presented in our paper with the evaluation script 'eval_model.py' as shown below. The parser arguments -t represents the tokenizer directory, -m the model directory, and -d the dataset directory respectively.

```
python eval_model.py -t ./Models/BERT -m ./Models/BERT -d ./Datasets/Arithmetic
python eval_model.py -t ./Models/BERT -m ./Models/BERT -d ./Datasets/Generalization/To1e5/Ours
python eval_model.py -t ./Models/BERT -m ./Models/BERT -d ./Datasets/Generalization/To1e6/Ours
python eval_model.py -t ./Models/BERT -m ./Models/BERT -d ./Datasets/Generalization/To1e7/Ours
```

```
python eval_model.py -t ./Models/BERT_Regression -m ./Models/BERT_Regression -d ./Datasets/Arithmetic
python eval_model.py -t ./Models/BERT_Regression -m ./Models/BERT_Regression -d ./Datasets/Generalization/To1e5/Ours
python eval_model.py -t ./Models/BERT_Regression -m ./Models/BERT_Regression -d ./Datasets/Generalization/To1e6/Ours
python eval_model.py -t ./Models/BERT_Regression -m ./Models/BERT_Regression -d ./Datasets/Generalization/To1e7/Ours
```

```
python eval_model.py -t ./Models/Skill-LM -m ./Models/Skill-LM -d ./Datasets/Arithmetic
python eval_model.py -t ./Models/Skill-LM -m ./Models/Skill-LM -d ./Datasets/Generalization/To1e5/Ours
python eval_model.py -t ./Models/Skill-LM -m ./Models/Skill-LM -d ./Datasets/Generalization/To1e6/Ours
python eval_model.py -t ./Models/Skill-LM -m ./Models/Skill-LM -d ./Datasets/Generalization/To1e7/Ours
```

The baseline uses AllenNLP and PyTorch to build their BERT-based variants, please refer to the author's repository [here](https://github.com/ag1988/injecting_numeracy) to run evaluations on these datasets.

## Training Skill-LM variants

The BERT and BERT_Regression models, as mentioned in the 'Repository Structure' section, can be trained with the following scripts:
> **Note:** Beginning the training sessions will overwrite the existing directories.

```
python train_bert.py
python train_bert_regression.py
```

Training our model, Skill-LM, however, requires the computation of the Fisher information matrix for the base BERT model first.

```
mkdir ./Fisher/Gradients
mkdir ./Fisher/Gradients/BERT

python ./Fisher/fisher_bert.py
```

This will populate the Fisher directory with the picklized Fisher matrix 'fisher_bert.pkl'. Now, we can train our model Skill-LM.

```
python train_bert_regression_ewc.py
```

## Evaluating Models on GLUE Tasks

To evaluate the models on the sets of GLUE tasks, a source installation of the Huggingface library is required. In a different environment, please run:

```
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
pip install -r ./examples/requirements.txt
```

With this environment, we can now train and evaluate our models on the set of GLUE tasks. To do so for other model variants, simply point --model_name_or_path to the respective model directory and change the output directory respectively.

```
mkdir ./GLUE
mkdir ./GLUE/CoLA
mkdir ./GLUE/QQP
mkdir ./GLUE/MNLI
mkdir ./GLUE/WNLI
mkdir ./GLUE/RTE
mkdir ./GLUE/MRPC
mkdir ./GLUE/STS-B
mkdir ./GLUE/SST-2
mkdir ./GLUE/QNLI

python run_glue.py --model_name_or_path ./Models/Skill-LM --task_name cola --max_seq_length 512 --do_train --do_eval --output_dir ./GLUE/CoLA --overwrite_output_dir --logging_steps 50
python run_glue.py --model_name_or_path ./Models/Skill-LM --task_name qqp --max_seq_length 512 --do_train --do_eval --output_dir ./GLUE/QQP --overwrite_output_dir --logging_steps 50
python run_glue.py --model_name_or_path ./Models/Skill-LM --task_name mnli --max_seq_length 512 --do_train --do_eval --output_dir ./GLUE/MNLI --overwrite_output_dir --logging_steps 50
python run_glue.py --model_name_or_path ./Models/Skill-LM --task_name wnli --max_seq_length 512 --do_train --do_eval --output_dir ./GLUE/WNLI --overwrite_output_dir --logging_steps 50
python run_glue.py --model_name_or_path ./Models/Skill-LM --task_name rte --max_seq_length 512 --do_train --do_eval --output_dir ./GLUE/RTE --overwrite_output_dir --logging_steps 50
python run_glue.py --model_name_or_path ./Models/Skill-LM --task_name mrpc --max_seq_length 512 --do_train --do_eval --output_dir ./GLUE/MRPC --overwrite_output_dir --logging_steps 50
python run_glue.py --model_name_or_path ./Models/Skill-LM --task_name stsb --max_seq_length 512 --do_train --do_eval --output_dir ./GLUE/STS-B --overwrite_output_dir --logging_steps 50
python run_glue.py --model_name_or_path ./Models/Skill-LM --task_name sst2 --max_seq_length 512 --do_train --do_eval --output_dir ./GLUE/SST-2 --overwrite_output_dir --logging_steps 50
python run_glue.py --model_name_or_path ./Models/Skill-LM --task_name qnli --max_seq_length 512 --do_train --do_eval --output_dir ./GLUE/QNLI --overwrite_output_dir --logging_steps 50
```

## Generating Plots

To replicate Figure 2 in the paper, we first need to build Fisher matrices for the BERT model trained for quantitative reasoning as well for the set of the 9 GLUE tasks. To build the Fisher matrices, we first need the pre-trained models. As we trained Skill-LM for the set of GLUE tasks above, we need to do the same for the base BERT model:

```
python run_glue.py --model_name_or_path bert-base-uncased --task_name cola --max_seq_length 512 --do_train --do_eval --output_dir ./GLUE/CoLA --overwrite_output_dir --logging_steps 50
python run_glue.py --model_name_or_path bert-base-uncased --task_name qqp --max_seq_length 512 --do_train --do_eval --output_dir ./GLUE/QQP --overwrite_output_dir --logging_steps 50
python run_glue.py --model_name_or_path bert-base-uncased --task_name mnli --max_seq_length 512 --do_train --do_eval --output_dir ./GLUE/MNLI --overwrite_output_dir --logging_steps 50
python run_glue.py --model_name_or_path bert-base-uncased --task_name wnli --max_seq_length 512 --do_train --do_eval --output_dir ./GLUE/WNLI --overwrite_output_dir --logging_steps 50
python run_glue.py --model_name_or_path bert-base-uncased --task_name rte --max_seq_length 512 --do_train --do_eval --output_dir ./GLUE/RTE --overwrite_output_dir --logging_steps 50
python run_glue.py --model_name_or_path bert-base-uncased --task_name mrpc --max_seq_length 512 --do_train --do_eval --output_dir ./GLUE/MRPC --overwrite_output_dir --logging_steps 50
python run_glue.py --model_name_or_path bert-base-uncased --task_name stsb --max_seq_length 512 --do_train --do_eval --output_dir ./GLUE/STS-B --overwrite_output_dir --logging_steps 50
python run_glue.py --model_name_or_path bert-base-uncased --task_name sst2 --max_seq_length 512 --do_train --do_eval --output_dir ./GLUE/SST-2 --overwrite_output_dir --logging_steps 50
python run_glue.py --model_name_or_path bert-base-uncased --task_name qnli --max_seq_length 512 --do_train --do_eval --output_dir ./GLUE/QNLI --overwrite_output_dir --logging_steps 50
```
Now, we compute the Fisher information matrices for these models:

```
python ./Fisher/fisher_bert_arith.py
python ./Fisher/fisher_bert_cola.py
python ./Fisher/fisher_bert_mrpc.py
python ./Fisher/fisher_bert_rte.py
python ./Fisher/fisher_bert_sst2.py
python ./Fisher/fisher_bert_stsb.py
python ./Fisher/fisher_bert_wnli.py
```

These scripts will populate the Fisher directory with the respective picklized Fisher matrices. With all these matrices in place, simply run the Plots-Fisher.ipynb to replicate the findings of Figure 2 and its continuations in the Appendix. Similarly, Error Plots.ipynb will replicate Figure 4.

**If our research or code aids your work, please cite us. Thanks!**

Contact: Mandar Sharma (mandarsharma@vt.edu)
