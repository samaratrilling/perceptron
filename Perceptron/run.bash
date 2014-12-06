#!/bin/bash

chmod +x q4.bash
chmod +x q5.bash
chmod +x q6.bash

mkdir output
echo "running question4."
date
./q4.bash
scp tag_dev.out output/

echo ""
echo "running question5."
date
./q5.bash
scp suffix_tagger.model output/
scp tag_dev_suffixes.out output/

echo ""
echo "running question6."
date
./q6.bash
scp experimental_tagger.model output/
scp tag_dev_experimental.out output/
echo "finished."
date
