#!/bin/bash

SPATTER_PATH=/home/user/tt-metal/build_Release/tt_metal/programming_examples/spatter/spatter
no_of_operations=$((2**24))
pattern_length=8

for (( j=0; j<=1; j++ ))
do
	if [[ $j == 0 ]]
	then
		echo -e "\n\n			Serial Version\n ********************************************************\n\n"
	else
		echo -e "\n\n			Parallel Version\n *******************************************************\n\n"
	fi
	
	for (( i=1; i<=8; i++ ))
	do
	  echo "Test No(Step size) : $i"
	  echo -e "\nGather version\n"
	  ${SPATTER_PATH} -p UNIFORM:${pattern_length}:$i -k gather -l ${no_of_operations} -b tt-metal -q 1 -i $j
	  echo -e "\nScatter version\n"
	  ${SPATTER_PATH} -p UNIFORM:${pattern_length}:$i -k scatter -l ${no_of_operations} -b tt-metal -q 1 -i $j
	  echo -e "\nMultiGather version\n"
	  ${SPATTER_PATH} -p UNIFORM:${pattern_length}:$i -k multigather -g UNIFORM:${pattern_length}:1 -l ${no_of_operations} -b tt-metal -q 1 -i $j
	  echo -e "\nMultiScatter version\n"
	  ${SPATTER_PATH} -p UNIFORM:${pattern_length}:$i -k multiscatter -u UNIFORM:${pattern_length}:1 -l ${no_of_operations} -b tt-metal -q 1 -i $j
	  echo -e "\nScatter_Gather version\n"
	  ${SPATTER_PATH} -u UNIFORM:${pattern_length}:$i -k sg -g UNIFORM:${pattern_length}:$i -l ${no_of_operations} -b tt-metal -q 1 -i $j
	done
done
