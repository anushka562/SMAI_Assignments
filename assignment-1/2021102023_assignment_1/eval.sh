#! /bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <test_data_file>"
    exit 1
fi

test_file_path=$1

python test.py "$test_file_path"
