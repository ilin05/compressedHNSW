#!/bin/bash
set -e

echo "Testing Ablation Differential Order"
../build/test_ablation_differential_order --dataset gist-960-euclidean fashion-mnist-784-euclidean mnist-784-euclidean sift-128-euclidean --algorithm DeXOR Elf Gorilla Camel