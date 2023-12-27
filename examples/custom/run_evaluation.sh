#!/bin/bash

# Define the model names to evaluate
model_names=("obs_12_1")
obs_types=("matrix")

for index in "${!model_names[@]}"; do
  model_name="${model_names[$index]}"
  obs_type="${obs_types[$index]}"

  # Call `cross_evaluate.py` with the current model name and options
  python cross_evaluate.py --model_name $model_name --num_battles 1 --save_replay --observation_type $obs_type
  python cross_evaluate.py --model_name $model_name --num_battles 100 --observation_type $obs_type

  # (Optional) Add additional logic for each iteration,
  # like logging results or performing analysis
  echo "Completed evaluation for model: $model_name"
done