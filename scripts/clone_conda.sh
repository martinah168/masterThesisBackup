#!/bin/bash

# old path as argument
old_path=$1

# store all directory names in old path in a variable
# and loop over them
dirs=$(ls $old_path)
echo dirs $dirs

for line in $dirs; do

    # extract name
    name=$line
    echo $name

    # construct path to environment
    path=$old_path/$name


    # skip line where name is base or empty or *
    if [[ $name == "base" || -z $name ]]; then
        continue
    fi

    echo "Cloning $name"
    echo "Path: $path"

    # create environment
    conda env remove -y -d -n $name # remove -d to actually remove existing env
    conda  create -y -d  --name $name --clone $path # remove -d to actually create new env

    break # remove this line to clone all environments

done
echo "Done"
