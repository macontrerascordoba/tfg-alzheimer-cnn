#! /bin/bash

rm condor_output/out/*
rm condor_output/err/*
rm condor_output/log/*

rm results/training/*
rm results/testing/*

rm *.pth

# Get the current date in the format yymmddHHMM
current_date=$(date +"%y%m%d%H%M")

# Start the seed
seed=1

first=true

models=(1 2 3)
learning_rate=(0.01 0.001 0.0001)
nfolds=(5 10)
patiences=(10 15)

# Overwrite the old file
#echo "" > task_params.txt

# Append the remaining lines to the file if needed
for model in "${models[@]}"; do
	for lrate in "${learning_rate[@]}"; do
		for fold in "${nfolds[@]}"; do
			for patience in "${patiences[@]}"; do
				if [ "$first" = true ] ; then
					echo "$seed $current_date $model $lrate $fold $patience" > task_params.txt
					first=false
				else
					echo "$seed $current_date $model $lrate $fold $patience" >> task_params.txt
				fi
			done
		done
	done
done

condor_submit task.sub

