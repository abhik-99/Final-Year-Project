# Final-Year-Project

## Authors
1. Abhik Banerjee.
2. Agniswar Roy.

## Problem Statement
miRNA Target Prediction by adapting it to Next Sentance Prediction and assessing the performance of SOTA NLP architectures on the problem. 

## Dataset
We are using the MBSTAR dataset which contains the human miRNA with the seed sequence from mRNA along with a score. This dataset is available at this [link](https://www.isical.ac.in/~bioinfo_miu/MBStar/MBStar_download20.htm)

The dataset has been divided into 2 parts for MLM and NSP tasks. The MLM partition is made larger with some of the sequences from NSP tasks also being available in this dataset. This will be used for training separate **[AlBERT](https://huggingface.co/transformers/model_doc/albert.html)** encoders on miRNA and mRNA sequence. 
The NSP downstream task will contain a custom NSP head where the output from these encoders will be fed. The NSP partition also has some sequences to test out the understanding of the encoders after the pre-training.