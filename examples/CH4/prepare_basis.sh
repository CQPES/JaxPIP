#!/bin/bash

# Step 1. Clone the MSA-2.0 repo and compile it

# clone repo
if [ ! -d "MSA-2.0" ]; then
    git clone https://github.com/mizu-bai/MSA-2.0.git
fi

# build
echo "Building msa executable..."
cd MSA-2.0/src/emsa
make
cd ../../../

# Step 2. Generate PIP basis for CH4 molecule
# The order of atoms is H H H H C
# The max degree of PIP is 4
# The basis will be stored in folder `basis`

echo "Generating PIP basis..."

# prepare
if [ ! -d "basis" ]; then
    mkdir basis
fi

cd basis
rm -rf ./*

# generate basis
# symtax: 
# ./msa [max degree] [molecule configuration]
# max degree = 4
# molecule configuration = 4 1
../MSA-2.0/src/emsa/msa 4 4 1

# Step 3. Convert BAS file to json
echo "Converting PIP basis to json format..."

# convert
python3 -u ../../../jaxpip/utils/BAS2json.py MOL_4_1_4.BAS

# rename
cp MOL_4_1_4.json A4B_4.json

echo "The PIP basis for A4B system with max degree up to 4 is now avaliable."
echo "The path to json format basis is basis/A4B_4.json."
