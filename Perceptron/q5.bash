#!/bin/bash

javac Q5.java
echo "Running q5 using tag_train.dat..."
java Q5
echo "Done training! Weights are in suffix_tagger.model."
javac Q5P2.java
echo "Running the perceptron algorithm with the weights from suffix_tagger.model..."
java Q5P2
echo "Done! Output available in tag_dev_suffixes.out."
