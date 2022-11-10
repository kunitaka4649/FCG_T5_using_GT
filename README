# README

This is the program for the system submitted to https://fcg.sharedtask.org/.
For FCG(feedback comment generation), we utilized grammatical terms(GTs).
We firstly predict GTs for each test instance, after that we add them into T5 input for FCG.
More details will be found on our paper(not public now).

## USAGE

Following this repo(https://github.com/kunitaka4649/FCG_GT_pred), you have to predict GTs for each test instance beforehand.

python 3.8.13

pip 22.1.2

```
pip -r requirements.txt
```

Please prepare public feedback comment dataset in accordance with
[Toward a Task of Feedback Comment Generation for Writing Learning](https://aclanthology.org/D19-1316) (Nagata, EMNLP 2019)
```
root
└─data
    └─train_dev
        ├─TRAIN.prep_feedback_comment.public.tsv
        ├─DEV.prep_feedback_comment.public.tsv
        └─TEST.prep_feedback_comment.public.tsv
```

### t5 baseline

```
train_baseline.sh <out_path> <seed> <gpu>
predict_baseline.sh <model_path> <seed> <gpu>
```

### t5 using GT

<pgt_train> is the path to predicted GTs for TRAIN data.
<pgt_dev> is the path to predicted GTs for DEV data.

```
train.sh <pgt_train> <pgt_dev> <out_path> <seed> <gpu>
predict.sh <pgt_dev> <model_path> <seed> <gpu>
```

For example,

```
train.sh trained_mdls/top-10/lr_3e-05.bin.train.result trained_mdls/top-10/lr_3e-05.bin.dev.result trained_mdls/t5_using_gt 0 0
predict.sh trained_mdls/top-10/lr_3e-05.bin.test.result trained_mdls/t5_using_gt 0 0
```