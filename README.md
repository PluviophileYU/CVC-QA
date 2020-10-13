# CVC-QA

## Counterfactual Variable Control for Robust and Interpretable Question Answering
This repository contains the code for the following paper:
* Sicheng Yu, Yulei Niu, Shuohang Wang, Jing Jiang, Qianru Sun *"Counterfactual Variable Control for Robust and Interpretable Question Answering (https://arxiv.org/abs/2010.05581)


## Requirement
* torch 1.3.1
* transformers 2.1.1
* apex 0.1
* tensorboardX 1.8
* prettytable 0.7.2

## Multiple-Choice Question Answering
Here we use RACE with BERT-base as example for MCQA task.

### Dowload data
- Step 1: Download original dataset via this link (http://www.cs.cmu.edu/~glai1/data/race/), and store them in directory `/data_mc/RACE`.
- Step 2: Download the adversarial sets via this link (https://drive.google.com/drive/folders/1ufPl0aP-QglVdsDtlKq9kTnt_0Fqmw2i?usp=sharing), and store them in same directory as step 1.

### CVC Training
```sh
cd src_mc
bash train.sh
```
You may visualize the loss trend using tensorboardX in directory `/src_mc/runs`.

### CVC-IV inference
Please change `--timestamp` according to your training time.
```sh
bash cvc_iv.sh
```

### CVC-MV inference (including training for c-adaptor)
Please change `--pre_model_dir` according to model selected by you.
```sh
bash cvc_mv.sh
```
### MCQA model trained by me
You can download the CVC model trained by us (CVC-MV is not included). You can find the results we reported in our paper. (https://drive.google.com/drive/folders/14ZMUwW_bxnpaDX4HbjdUxGwcNBzFIYR6?usp=sharing) 

## Span-Extraction Question Answering
Coming Soon!
