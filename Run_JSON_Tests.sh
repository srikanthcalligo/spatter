#!/bin/bash

json_file_name=$1
compute_mode=$2
parallel_mode=$3
verbose=$4
python3 - << EOF
import json
import os

with open("${json_file_name}", "r") as fp:
    lines = json.load(fp)

for line in lines:
    with open("json_input_file.json", "w") as fop:
        fop.write("[")
        json.dump(line, fop, separators=(', ', ':'))
        fop.write("]")
    os.system("/home/user/tt-metal/build/tt_metal/programming_examples/spatter/spatter -f json_input_file.json -v ${verbose} -q ${compute_mode} -i ${parallel_mode}")
EOF
