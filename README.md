# CVC-QA

## Counterfactual Variable Control for Robust and Interpretable Question Answering
This repository contains the code for the following paper:
* Sicheng Yu, Yulei Niu, Shuohang Wang, Jing Jiang, Qianru Sun *"Counterfactual Variable Control for Robust and Interpretable Question Answering"*

## Requirement
* torch 1.3.1
* transformers 2.1.1
* apex 0.1
* tensorboardX 1.8
* prettytable 0.7.2

## Multiple-Choice Question Answering
Here we use RACE with BERT-base as example for MCQA task.

### Dowload data
- Step 1: Download original dataset via this link (http://www.cs.cmu.edu/~glai1/data/race/), and store in directory `/data_mc/RACE`
- Step 2: Download the adversarial sets created by ours via this link ()
