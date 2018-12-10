import numpy as np
import sys
if "../" not in sys.path:
  sys.path.append("../")
from lib.envs.gridworld import GridworldEnv
env = GridworldEnv()


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for i in range(env.nS):
            v = V[i]
            w=0
            for j in range(env.nA):
                print(env.P[i][j][0])
                w+= policy[i][j] * env.P[i][j][0][0] * (env.P[i][j][0][2] + discount_factor * V[env.P[i][j][0][1]])
            V[i]=w
            if delta < abs(v - V[i]):
                delta = abs(v - V[i])

        if delta < theta:
            break
        # TODO: Implement!
            """
            solution:
           
            delta=0
            for s in range(env.nS):
            v=0
                for a,action_prob in enumerate(policy[s])
                    for prob,next_state,reward,done in env.P[s][a]:
                    v+=action_prob*prob*(reward+discount*V[next_state])
                delta=max(delta,np.abs(v-V[s]))
                v[s]=v
            if delta<theta:
                break
            
            """
    return np.array(V)
random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)
# Test: Make sure the evaluated policy is what we expected
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
print(v)