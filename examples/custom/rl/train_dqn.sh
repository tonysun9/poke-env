#!/bin/bash

# Function to define a training run with default values
train_model() {
  local lr=0.00001
  local model_name="default_model"
  local batch_size=256
  local buffer_size=50000
  local exploration_fraction=1.0
  local learning_start=0
  local train_steps=1000000
  local self_play="random"

  # Loop through arguments and process them
  while [ $# -gt 0 ]; do
      case "$1" in
          --lr)
              lr="$2"
              shift 2
              ;;
          --model_name)
              model_name="$2"
              shift 2
              ;;
          --batch_size)
              batch_size="$2"
              shift 2
              ;;
          --buffer_size)
              buffer_size="$2"
              shift 2
              ;;
          --exploration_fraction)
              exploration_fraction="$2"
              shift 2
              ;;
          --learning_start)
              learning_start="$2"
              shift 2
              ;;
          --train_steps)
              train_steps="$2"
              shift 2
              ;;
            --self_play)
              self_play="$2"
              shift 2
              ;;
          *)
              break
              ;;
      esac
  done

  echo "Training run with parameters:"
  echo "Learning rate: $lr"
  echo "Model name: $model_name"
  echo "Batch size: $batch_size"
  echo "Buffer size: $buffer_size"
  echo "Exploration fraction: $exploration_fraction"
  echo "Learning start: $learning_start"
  echo "Train steps: $train_steps"
  echo "Self play: $self_play"
  # Add the command to start training here, using the above variables

  # Call `sb_observation.py` with current parameters
  python sb_observation.py \
    --learning_rate $lr \
    --model_name $model_name \
    --batch_size $batch_size \
    --buffer_size $buffer_size \
    --exploration_fraction $exploration_fraction \
    --learning_start $learning_start \
    --train_steps $train_steps \
    --self_play $self_play

  echo "Completed run with learning rate: $lr, batch size: $batch_size, " \
    "buffer_size: $buffer_size, model name: $model_name, " \
    "exploration_fraction: $exploration_fraction, train_steps: $train_steps" \
    "self_play: $self_play"
}

# train_model \
#   --lr 0.00003 \
#   --model_name "model_name" \
#   --batch_size 256 \
#   --buffer_size 50000 \
#   --train_steps 100000 \
#   --self_play "random"

train_model \
  --model_name "obs_13_1" \
  --self_play "random"

train_model \
  --model_name "obs_13_2" \
  --self_play "obs_13_1" \

train_model \
  --model_name "obs_13_3" \
  --self_play "obs_13_2" \

# train_model \
#   --model_name "obs_10_3" \
#   --self_play "obs_10_2"

# train_model \
#   --model_name "obs_10_4" \
#   --self_play "obs_10_3"

# train_model \
#   --model_name "obs_9_5" \
#   --self_play "obs_9_4"

# train_model \
#   --model_name "obs_9_6" \
#   --self_play "obs_9_5"