# PAGE: A Position-Aware Graph-based model for Emotion cause entailment
The source code for paper(appeared in EMNLP 2023) -- PAGE: A POSITION-AWARE GRAPH-BASED MODEL FOR EMOTION CAUSE ENTAILMENT IN CONVERSATION.
## Environment
- Python==3.7.11
- Pytorch==1.10.2
- Transformers==4.6.0
- Torch-geometric==2.1.0.post1
## Data
Our model uses the conversation dataset [RECCON](https://github.com/declare-lab/RECCON/tree/main/data/subtask2/fold1), which consists of two subsets: RECCON-DD and RECCON-IE.  
We follow [KEC](https://github.com/LeqsNaN/KEC), the processed dataset are in the `./data` directory.
## Run
We use pre-trained language model RoBERTa for our experiments.
```
python main.py
```
