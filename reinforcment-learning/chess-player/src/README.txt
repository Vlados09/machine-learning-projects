Simply running chess_student.py will run a Q-learning algorithm with optimal parameters and henormal initialisation. 
With such a set up it takes under 10 minutes to run a single iteration of the code. Running my_test.py will partially reproduce the results. 
The number of repetitions is set to 0 meaning the code is run only once, thus producing more noisier version of results. 
To display graph with zoom on final episodes line 239 has to be uncommented, but may require installing mpl_toolkits. 
It is also possible to delete SARSA_0 parameter to just see the comparison between standard SARSA and Q-learning. 
Lines 236 and 237 if uncomented would reproduce results for beta and gamma respectively. 
The repetitions are set to 0 again but can be set to 6 in order to reproduce results presented in the figures exactly. (May take long time to run...). 
It is also possible to change the values of tested parameters by editing the list passed in. 
Lines 234 was used to test different initialisation techniques and can be sped up by passing optimal parameters. 
Line 235 was used to produce results with solution for exploding weights. 
Most personal code was written in my_code file with few exceptions such as error calculation.