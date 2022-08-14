# LSVI-UCB
An implementation of the algorithm presented in the paper 
[Provably Efficient Reinforcement Learning with Linear Function Approximation](https://arxiv.org/pdf/1907.05388.pdf).  
The code was inspired by Alex Ayoub's repo [Exploration-in-RL](https://github.com/aa14k/Exploration-in-RL).
### Prerequisites
```
python > 3.8
numpy==1.23.1
```

### MDP - Deterministic RiverSwim

![alt text](https://www.researchgate.net/publication/357201959/figure/fig1/AS:1103292767711237@1640056902545/RiverSwim-MDP-solid-and-dotted-arrows-denote-the-transitions-under-actions-right-and.ppm)

Where the actions are:  
0 - Stay on the same state  
1 - Move right  

## Experiments
- State-action values of Q(s, a=1) for different chain lengths.

1. Chain length: 4  
![Alt text](artifacts/q_4.png)
2. Chain length: 5  
![Alt text](artifacts/q_5.png)
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
