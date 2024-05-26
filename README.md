# RL-INTRO
Python code implementation of the introduction to Reinforcement Learning book

## Dynamic Programming

### Equiprobable policy evaluation for 4 by 4 grid
`dynamic_programming/policy_evaluation_4_by_4_grid.py` implements policy evaluation of an equiprobable action selection for `environments/gridworld.py` environment.
The environment is shown in the screenshot below.

![alt text](/assets/4_by_4_grid.png)

The pseudocode for the policy evaluation is shown below.

![alt text](/assets/4_by_grid_policy_evaluation.png)

With a `θ = 0.001`, the policy evaluation values stop improving after about 130 steps with values very close to the optimum values when we choose to do infinite iteration.

Below are the final state values after 131 iterations.

|     |1      |2     |3       |4     |
|-----|-------|:-----|:-------|:-----|
|**1**|  0.   |-13.99 |-19.98 |-21.98|
|**2**|-13.99 |-17.99 |-19.98 |-19.98|
|**3**|-19.98 |-19.98 |-17.99 |-13.99|
|**4**|-21.98 |-19.98 |-13.99 |  0.  |

The table below shows optimal state values after infinite iterations

|     |1   |2   |3   |4  |
|-----|----|:---|:---|:--|
|**1**| 0  |-14 |-20 |-22|
|**2**|-14 |-18 |-20 |-20|
|**3**|-20 |-20 |-18 |-14|
|**4**|-22 |-20 |-14 |0  |

With `θ = 0.001`, the state values are optimum after 173 steps. The results are even better with inplace value update, optimals state values were found after only 114 steps.
