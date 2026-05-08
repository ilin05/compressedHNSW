#!/bin/bash
set -e

echo "Testing Ablation Chain Length"
../build/test_ablation_chain_length --dataset gist-960-euclidean fashion-mnist-784-euclidean mnist-784-euclidean sift-128-euclidean --algorithm DeXOR Elf Gorilla Camel --chain_max 2 5 -1