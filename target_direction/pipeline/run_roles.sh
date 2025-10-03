#!/bin/bash

# Array of roles
roles=(
  "bailiff"
  "lawyer"
  "data analyst"
  "mathematician"
  "statistician"
  "nurse"
  "doctor"
  "physician"
  "dentist"
  "surgeon"
  "geneticist"
  "biologist"
  "physicist"
  "teacher"
  "chemist"
  "ecologist"
  "politician"
  "sheriff"
  "enthusiast"
  "partisan"
  "psychologist"
)

# Model path
MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"

# Loop through each role and execute the pipeline
for role in "${roles[@]}"; do
    echo "Processing role: $role"
    python -m pipeline.run_pipeline --model_path "$MODEL_PATH" --role "$role"
    
done

echo "All roles processed successfully!"
