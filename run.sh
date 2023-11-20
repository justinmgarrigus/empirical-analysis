#!/bin/bash 

gcc -DCONWAYLIFE -DITERATIONS=25 -DEXECUTIONS=1000 automata.c -o automata
./automata conwaylife-25-1000.txt

gcc -DLIFEWITHOUTDEATH -DITERATIONS=25 -DEXECUTIONS=1000 automata.c -o automata
./automata lifewithoutdeath-25-1000.txt 

gcc -DDAYANDNIGHT -DITERATIONS=25 -DEXECUTIONS=1000 automata.c -o automata 
./automata dayandnight-25-1000.txt 
