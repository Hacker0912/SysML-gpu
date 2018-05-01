cpu_dir=./cpu_results/
gpu_dir=./gpu_results/
mkdir ${cpu_dir}
mkdir ${gpu_dir}

# GPU without fit in memory:
echo "Running GPU experiment without fitting in memory ..."
for method in logistic_regression least_square linear_svm
do
  #for m in 1000000 10000000 2000000 5000000
  for m in 20000000 50000000
  do
  	  echo "Running ${method} on T_${m}"
      python test.py \
	    ${method} \
	    --optim=SCD \
	    --data-dir=./data/T_${m} \
	    --num-tuples=${m} \
	    --num-feats=20 \
	    --max-steps=50 \
	    --lr=0.5 \
	    --batch-size=1 \
	    --load-in-memory= \
	    --enable-gpu=True > ${gpu_dir}GPU_${method}_T_${m}_without_fit 2>&1
	done
done

# GPU fit in memory:
echo "Running GPU experiment fitting in memory ..."
for method in logistic_regression least_square linear_svm
do
  #for m in 1000000 10000000 2000000 5000000
  for m in 20000000 50000000  
  do
  	  echo "Running ${method} on T_${m}"
      python test.py \
	    ${method} \
	    --optim=SCD \
	    --data-dir=./data/T_${m} \
	    --num-tuples=${m} \
	    --num-feats=20 \
	    --max-steps=50 \
	    --lr=0.5 \
	    --batch-size=1 \
	    --load-in-memory=True \
	    --enable-gpu=True > ${gpu_dir}GPU_${method}_T_${m}_fitting 2>&1
	done
done

# CPU without fit in memory:
echo "Running CPU experiment without fitting in memory ..."
for method in logistic_regression least_square linear_svm
do
  #for m in 1000000 10000000 2000000 5000000
  for m in 20000000 50000000
  do
  	  echo "Running ${method} on T_${m}"
      python test.py \
	    ${method} \
	    --optim=SCD \
	    --data-dir=./data/T_${m} \
	    --num-tuples=${m} \
	    --num-feats=20 \
	    --max-steps=50 \
	    --lr=0.5 \
	    --batch-size=1 \
	    --load-in-memory= \
	    --enable-gpu= > ${cpu_dir}CPU_${method}_T_${m}_without_fit 2>&1
	done
done

# CPU with fit in memory:
echo "Running CPU experiment fitting in memory ..."
for method in logistic_regression least_square linear_svm
do
  #for m in 1000000 10000000 2000000 5000000
  for m in 20000000 50000000
  do
  	  echo "Running ${method} on T_${m}"
      python test.py \
	    ${method} \
	    --optim=SCD \
	    --data-dir=./data/T_${m} \
	    --num-tuples=${m} \
	    --num-feats=20 \
	    --max-steps=50 \
	    --lr=0.5 \
	    --batch-size=1 \
	    --load-in-memory=True \
	    --enable-gpu= > ${cpu_dir}CPU_${method}_T_${m}_fitting 2>&1
	done
done