#!/usr/bin/env bash

wget --no-check-certificate https://github.com/microsoft/CodeT/raw/main/RepoCoder/datasets/datasets.zip -O datasets.zip
unzip datasets.zip -d datasets
wget --no-check-certificate https://github.com/microsoft/CodeT/raw/main/RepoCoder/repositories/function_level.zip -O function_level_repositories.zip
unzip function_level_repositories.zip -d repositories
wget --no-check-certificate https://github.com/microsoft/CodeT/raw/main/RepoCoder/repositories/line_and_api_level.zip -O line_and_api_level_repositories.zip
unzip line_and_api_level_repositories.zip -d repositories
