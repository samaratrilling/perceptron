#!/bin/bash

javac Q6.java
echo "Running q6 using tag_train.dat..."
java Q6
echo "Done training! Weights are in experimental_tagger.model."
javac Q6P2.java
echo "Running the perceptron algorithm with the weights from experimental_tagger.model."
java Q6P2
echo "Done! Output available in tag_dev_experimental.out"
