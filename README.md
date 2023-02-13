# SBERTiment: A New Pipeline to Solve Aspect Based Sentiment Analysis in the Zero-Shot Setting

In this repo you can find the code to reproduce the experiments described in the
paper **SBERTiment: A New Pipeline to Solve Aspect Based Sentiment Analysis in the
Zero-Shot Setting**.

First of all you need to install the python packages needed to train and
evaluate models. You can do that with the following command:

`pip3 install -r requirements.txt`

To run all the experiments described in the paper it is sufficient to launch a
shell script:

`./run.sh`

When this program ends, you will find in the folder `data/results/` a series of
`.csv` files containing the results of the experiments.

In folders `acsa_add_one`, `seq2seq` and `our_pipeline` you will find all code
needed to run experiments with *AddOneDim-BERT*, *Seq2seq* and *Our pipeline*
methods described in the paper. In particular, you will find all
classes for prediction together with scripts to generate specific train datasets
and launch trainings.  
