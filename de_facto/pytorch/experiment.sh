# GPU without fit in memory:
echo "Running GPU experiment without fitting in memory ..."
for method in logistic_regression least_square linear_svm
do
  for file in T_1000000 T_10000000 T_2000000 T_5000000
  do
  	  echo "Running ${method} on ${file}"
      python test.py \
	    ${method} \
	    --optim=SCD \
	    --data-dir=./data/${file} \
	    --num-tuples=10000000 \
	    --num-feats=20 \
	    --max-steps=50 \
	    --lr=0.5 \
	    --batch-size=1 \
	    --load-in-memory= \
	    --enable-gpu=True > GPU_${method}_${file}_without_fit
	done
done

# GPU fit in memory:
echo "Running GPU experiment fitting in memory ..."
for method in logistic_regression least_square linear_svm
do
  for file in T_1000000 T_10000000 T_2000000 T_5000000
  do
  	  echo "Running ${method} on ${file}"
      python test.py \
	    ${method} \
	    --optim=SCD \
	    --data-dir=./data/${file} \
	    --num-tuples=10000000 \
	    --num-feats=20 \
	    --max-steps=50 \
	    --lr=0.5 \
	    --batch-size=1 \
	    --load-in-memory=True \
	    --enable-gpu=True > GPU_${method}_${file}_fitting
	done
done

# CPU without fit in memory:
echo "Running CPU experiment without fitting in memory ..."
for method in logistic_regression least_square linear_svm
do
  for file in T_1000000 T_10000000 T_2000000 T_5000000
  do
  	  echo "Running ${method} on ${file}"
      python test.py \
	    ${method} \
	    --optim=SCD \
	    --data-dir=./data/${file} \
	    --num-tuples=10000000 \
	    --num-feats=20 \
	    --max-steps=50 \
	    --lr=0.5 \
	    --batch-size=1 \
	    --load-in-memory= \
	    --enable-gpu= > CPU_${method}_${file}_without_fit
	done
done