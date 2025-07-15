#!/bin/bash
# [p, ) excluding turn_switch
env_dir="/home/fyc/RoboTwin/envs"
count=0

for file in "$env_dir"/*.py; do
    filename=$(basename "$file" .py)
    first_char=${filename:0:1}

    if [[ "$filename" != _* && "$first_char" =~ [p-z] && "$filename" != "turn_switch" ]]; then
        echo "$filename"
        bash collect_data.sh $filename demo_randomized 3
        ((count++))
    fi
done

echo "Total tasks processed: $count"
