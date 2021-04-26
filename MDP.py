# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 18:10:30 2021

@author: Joey
"""

import gym
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import hiive.mdptoolbox 
import hiive.mdptoolbox.example
import hiive.mdptoolbox.mdp
import time


def get_reward(env, policy, g):
    observed = env.reset()
    total = 0
    index = 0
    step_list = []
    while True:
#        env.render()
        observed, reward, stop_, _ = env.step(int(policy[observed]))
        total += (g ** index * reward)
        index += 1
        if stop_:
            step_list.append(index)
            break
    return total

def policy_eval(env, policy, g, n):
    
    return np.mean([get_reward(env, policy, g) for _ in range(n)])

def multi_policy_eval(env, multi_policy, g, n):
    score = []
    for policy in multi_policy:
        score_policy = []
        for seed in range(10):
            env.seed(seed)
            scoring_seed = [get_reward(env, policy, g) for _ in range(n)]
            score_policy.append(np.mean(scoring_seed))
        score.append(np.mean(score_policy))
    return score

def compute_v(env, g, policy):
    s1 = np.zeros((env.nS, env.nS)) 
    s2 = np.zeros(env.nS)
    for s in range(env.nS):
        
        for pol_,state_,rew_, is_done in env.P[s][policy[s]]:
            s2[s] += pol_*rew_
            s1[s, state_] += pol_ * gamma
    s1 -= np.eye(env.nS)
    
    return -np.linalg.solve(s1, s2).T

def compute_q(env, g, v):
    q = np.zeros([env.nS,env.nA])
    for state in range(env.nS):
        for action in range(env.nA):
            for pol_, state_, rew_, is_done in env.P[state][action]:
                q[state, action] += pol_ * (rew_ +  g * v[state_]) 
                
    return q
    
def policy_iteration(env, g):
     # POLICY ITERATION
     
    policy_list = []
    diff_vals = []
    avg_values = []
    time_itr = []
    g = g
    print(g)
    prev_policy = np.zeros(env.nS)
    k_value = 0
    val = [np.zeros(env.nS)]
    for i in range(50000):
        prev_val = val[-1]
        
        eps  = .000000000001
        start_time = time.time()
#        while True:
#            prev_val = np.copy(val)
#            for x in range(env.nS):
#                policy_1 = policy[x]
#                val[x] = sum([pol_ * (rew_ +  g * prev_val[state_]) for pol_, state_, rew_, is_done in env.P[x][policy_1]])
#            if(np.sum((np.fabs(prev_val - val))) <= eps):
#                break
        
        og_policy = compute_v(env, g, prev_policy)
        new_policy = compute_q(env, g, og_policy)
        policy = new_policy.argmax(axis = 1)
        time_itr.append(time.time() - start_time)
        diff_vals.append(np.sum((np.fabs(og_policy - prev_val))) )
        avg_values.append(np.mean(og_policy))
        
        val.append(og_policy)
        policy_list.append(policy)
        prev_policy = policy
        x = (i > 0)
        y = (np.sum((np.fabs(og_policy - prev_val))) <= eps)

        if((x and y) == True):
            return val, policy_list, diff_vals,avg_values , i, time_itr
            
        
        mean_score_list = np.mean([get_reward(env, policy, g) for _ in range(10)])   
        
        
    return val, policy_list, diff_vals,avg_values , i+1, time_itr
def value_iteration(env, g):
      # VALUE IERATION
    
    policy_list = []
    diff_vals = []
    avg_values = []
    time_itr = []
    g = g
    val_list  = [np.zeros(env.nS)]
    eps = 1e-10
    k_value = 0
    for i in range(2000):
        pre_val = val_list[-1]
        start_time = time.time()
        vals = np.copy(pre_val)
        policy = np.zeros(env.nS)

        for state in range(env.nS):
            
            val = vals[state]
            action = policy[state]
            for a in range(env.nA):   
                val_a = np.sum([pol_ * (rew_ +  g * pre_val[state_]) for pol_, state_, rew_, is_done in env.P[state][a]])
                if val_a > val:
                    val = val_a
                    action = a
            
            vals[state] = val
            policy[state] = action
        time_itr.append(time.time() - start_time)    
        diff_vals.append(np.sum((np.fabs(vals - pre_val))))
        val_list.append(vals)
        policy_list.append(policy)
        avg_values.append(np.mean(vals))
        
        
        if(np.sum((np.fabs(vals - pre_val))) <= eps):
            print(i)
            return vals, val_list, policy_list, diff_vals,avg_values , i, time_itr

    
    
    
    
    mean_score_list = np.mean([get_reward(env, policy, g) for _ in range(10)])   
    return vals, val_list, policy_list, diff_vals,avg_values, i, time_itr

def choose_action(observation, epsi):
    action = 0
    if np.random.uniform(0, 1) < epsi:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[observation, :])
    return action

def Q_learning(env):
    epsi_list = [0.05 , 0.5 , 0.9 ]
    q_table  = np.zeros((env.observation_space.n, env.action_space.n))
    total_reward =[]
    total_iter = []
    avg_array = []
    split_array = []
    size_array = []
    g = 0.95
    learning_rate = 0.90
    episodes = 10000
    fl_environment = 'FrozenLake-v0'
    env = gym.make(fl_environment).unwrapped
    for epsi in epsi_list:
        rewards = []
        iter_list = []
        for epi in range(episodes):  
            episode_reward = 0
            state = env.reset()
            t = 0
            max_steps = 5000
            done = False
            for i in range(max_steps):        
                if done:
                    break
                current = state
                if np.random.rand() < epsi:
                    a = np.argmax(q_table[current, :])
                else:
                     a = env.action_space.sample()
        
                state, r, done, info = env.step(a)
                t = t + r
                q_table[current, a] += learning_rate * (r + g * np.max(q_table[state, :]) - q_table[current, a])
            epsi = 1 - 2.71 **(-epsi / 1000)
            rewards.append(t)
            iter_list.append(i)
            total_reward.append(rewards)
            total_iter.append(iter_list)
    
        def split_list(my_list, n_pieces):
            for x in range(0, len(my_list), n_pieces):
                yield my_list[i:i+n_pieces]
            
            
        split_rew = list(split_list(rewards, int(episodes/ 50)))
        mean_split = [sum(x) / len(x) for x in split_rew]
        avg_array.append(mean_split)
        split_array.append(int(episodes/ 50))
        size_array.append(int(episodes/ 50))
    
    
    plt.plot(range(0, len(total_reward[0]), size_array[0]), avg_array[0])
    plt.show()           
    return(q_table, total_reward, total_iter)
                
            
            

    
def frozen_lake_MPD():
    
    fl_environment = 'FrozenLake-v0'
    
    env = gym.make(fl_environment).unwrapped
    gamma_list = []
    iter_list = []
    avg_val =[]
    avg_time = []
    for i in range(0, 10):
        gamma = (i + 0.10) / 10
        gamma_list.append(gamma)
        print(gamma)
        values_policy, policy_list_policy, diff_val_policy, avg_vals_policy, iterations_policy, time_itr_policy = policy_iteration(env, gamma)
        iter_list.append(iterations_policy)
        policy_sublist_p = [policy_list_policy[i] for i in np.linspace(0, len(policy_list_policy)-1, 25, dtype='int')]
        score_v = multi_policy_eval(env, policy_sublist_p, g, 100)
        avg_val.append(np.mean(avg_vals_policy))
        avg_time.append(np.mean(time_itr_policy))
        
    
    plt.plot(gamma_list, iter_list)
    plt.title("Gamma vs Iterations")
    plt.xlabel("gamma")
    plt.ylabel("iterations")
    plt.show()
    
    plt.plot(gamma_list, avg_val)
    plt.title("Gamma vs Mean_value")
    plt.xlabel("gamma")
    plt.ylabel("Average Value")
    plt.show()
    
    plt.plot(range(0,(iter_list[-1] + 1)), diff_val_policy)
    plt.title("Iterations vs Difference")
    plt.xlabel("Iterations")
    plt.ylabel("Difference")
    plt.show()
    
    plt.plot(range(0, len(avg_time)), avg_time)
    plt.xlabel("Iterations")
    plt.ylabel("Time (ms)")
    plt.show()
    
    
    fl_environment = 'FrozenLake-v0'
    
    env = gym.make(fl_environment).unwrapped
    gamma_list = []
    iter_list = []
    avg_val =[]
    avg_time = []
    for i in range(0, 10):
        gamma = (i + 0.10) / 10
        gamma_list.append(gamma)
        values, value_list, policy_list, diff_val, avg_vals, iterations, time_itr = value_iteration(env, gamma)
        iter_list.append(iterations)
        policy_sublist_v = [policy_list[i] for i in np.linspace(0, len(policy_list)-1, 25, dtype='int')]
        score_v = multi_policy_eval(env, policy_sublist_v, g, 100)
        avg_val.append(np.mean(avg_vals))
        avg_time.append(np.mean(time_itr))
        
        
    plt.plot(gamma_list, iter_list)
    plt.title("Gamma vs Iterations")
    plt.xlabel("gamma")
    plt.ylabel("iterations")
    plt.show()
    
    plt.plot(gamma_list, avg_val)
    plt.title("Gamma vs Mean_value")
    plt.xlabel("gamma")
    plt.ylabel("Average Value")
    plt.show()
    
    plt.plot(range(0,(iter_list[-1] + 1)), diff_val)
    plt.title("Iterations vs Difference")
    plt.xlabel("Iterations")
    plt.ylabel("Difference")
    plt.show()
    
    plt.plot(range(0, len(avg_time)), avg_time)
    plt.xlabel("Iterations")
    plt.ylabel("Time (ms)")
    plt.show()
    
    
    Q_table, Rewards, iters_Q = Q_learning(env)
    avg_it_Q = []
    for i in iters_Q:
        avg_it_Q.append(np.mean(i))
    plt.plot(range(0, 30000),avg_it_Q)
    plt.title("Average Iterations before reaching Success")
    
def forest_MPD():
    
    
        
    
    
        
          
    prob_matrix, reward_matrix =  hiive.mdptoolbox.example.forest(S = 5000, p = 0.05)
    q_table = []
    reward = []
    
    max_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.005
    epsi = max_epsilon
    g = 0.80
    ql = hiive.mdptoolbox.mdp.QLearning(prob_matrix, reward_matrix, 0.95, epsilon_decay = 0.90)  
    ql.run()    
    
    
    q_rew = []
    for i in ql.run():
        q_rew.append(i["Reward"])
    
    
    
    policy_iteration = hiive.mdptoolbox.mdp.PolicyIteration(prob_matrix, reward_matrix, 0.95)  
    pi = policy_iteration.run()
            
            
            
            
            
            
            
            