#!/bin/bash

# Define default parameters
env_name="simple_spread_v3"
episode_num=3000
episode_length=3
agent_num=80  # Default agent number

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --env_name) env_name="$2"; shift; shift ;;
    --episode_num) episode_num="$2"; shift; shift ;;
    --episode_length) episode_length="$2"; shift; shift ;;
    --learn_interval) learn_interval="$2"; shift; shift ;;
    --random_steps) random_steps="$2"; shift; shift ;;
    --tau) tau="$2"; shift; shift ;;
    --gamma) gamma="$2"; shift; shift ;;
    --buffer_capacity) buffer_capacity="$2"; shift; shift ;;
    --batch_size) batch_size="$2"; shift; shift ;;
    --actor_lr) actor_lr="$2"; shift; shift ;;
    --critic_lr) critic_lr="$2"; shift; shift ;;
    --agent_num) agent_num="$2"; shift; shift ;;  # Optional argument
    *) echo "Unknown option $1"; exit 1 ;;
  esac
done

#First run with the first set of parameters
#echo "Running first instance with default parameters..."
#python3 main.py \
#  "$env_name" \
#  --episode_num $episode_num \
#  --episode_length $episode_length \
#  --agent_num $agent_num

# Modify parameters for the second run
env_name="simple_spread_v2"
episode_num=3000
agent_num=80
episode_length=100

# Second run with modified parameters
echo "Running second instance with modified parameters..."
python3 main.py \
  "$env_name" \
  --episode_num $episode_num \
  --episode_length $episode_length \
  --agent_num  $agent_num
