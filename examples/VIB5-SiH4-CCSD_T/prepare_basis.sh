#!/bin/bash

rm -rf MSA-2.0 basis

git clone https://github.com/mizu-bai/MSA-2.0.git
cd MSA-2.0/src/emsa
make
cd ../../../
mkdir basis
cd basis
# order = 4
# SiH4 -> Si 1 H 4 -> 1 4
../MSA-2.0/src/emsa/msa 4 1 4

python3 -u ../../../jaxpip/utils/BAS2json.py MOL_1_4_4.BAS
