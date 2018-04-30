python test.py \
linear_svm \
--optim=SCD \
--data-dir=./data/T_float \
--num-tuples=10000000 \
--num-feats=20 \
--max-steps=50 \
--lr=0.5 \
--batch-size=1 \
--enable-gpu=True \
--load-in-memory=False