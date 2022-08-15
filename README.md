# LSVI-UCB
An implementation of the algorithm presented in the paper 
[Provably Efficient Reinforcement Learning with Linear Function Approximation](https://arxiv.org/pdf/1907.05388.pdf).  
The code was built on top of Alex Ayoub's repo [Exploration-in-RL](https://github.com/aa14k/Exploration-in-RL).  

### Important notes
1. The algorithm is very sensetive to the choise of beta  
2. Only when we added epsilon greedy we managed to get decent results  

### Prerequisites
```
python > 3.8
numpy==1.23.1
```

### MDP - RiverSwim

![alt text](artifacts/8A45B9DB-4DDD-40A0-B0FD-444E976A9CFD.png)
  
A **deterministic** 1D state cahin with two possible actions:  
0 - Stay on the same state  
1 - Move right  
The rewards are 0 for all 0 actions (r(s,a=0)=0), a small negative reward ("-k" in the drawing, order of 0.1 or 0.01) for 1 actions in all states except of the state before the terminal state, and reward of 1 transitioning to the terminal state.  
The green marks in the drawing above are the reward of the MDP.


## Experiments
- State-action values of Q(s, a=1) for different chain lengths.

1. Chain length: 4  
![Alt text](artifacts/q_4.png)  
For earlier states it takes more steps to reach their true value.  
2. Chain length: 5  
![Alt text](artifacts/q_5.png)  
It takes more steps for longer chains to converge.  
3. Chain length: 6  
![Alt text](artifacts/q_6.png)

- For Q(s, a=0), the algorithm failed to reach 0 value for all states:  
![Alt text](artifacts/q_4_a0.png)
  
- Rewards  
1. Chain length: 4  
![Alt text](artifacts/reward_4.png)  
Zoom in on the early steps:  
![Alt text](artifacts/reward_4_2000.png)
2. Chain length: 5  
![Alt text](artifacts/reward_5.png)  
If we reduce the negative reward in a factor of 10:  
![Alt text](artifacts/reward_5_small.png)


### Next steps  
experiment with other VI based algorithms.  
