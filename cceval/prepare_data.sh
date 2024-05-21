#!/usr/bin/env bash


git clone https://github.com/amazon-science/cceval.git
mv cceval/data/* .
rm -rf cceval

mkdir -p processed_data
tar -xvJf crosscodeeval_data.tar.xz -C processed_data
