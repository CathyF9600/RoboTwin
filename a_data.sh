#!/bin/bash
# [a, p)
env_dir="/home/fyc/RoboTwin/envs"
count=0

for file in "$env_dir"/*.py; do
    filename=$(basename "$file" .py)
    first_char=${filename:0:1}

    if [[ "$filename" != _* && "$first_char" =~ [b-o] ]]; then
        echo "$filename"
        bash collect_data.sh $filename demo_randomized 7
        ((count++))
    fi
done

echo "Total tasks processed: $count"
