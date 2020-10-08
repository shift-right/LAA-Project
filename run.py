from LAA_env import LAA
from DQN_Brain import DeepQNetwork
from DoubleDQN_Brain import DoubleDQN
from DuelingDQN_Brain import DuelingDQN
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import numpy as np
import time

#Train
#------------------------------------------------------------------------------------------#
def run_LAA():
    
    count = 0
    step = 0
    reward_total_ql = 0
  
    for episode in range(100):
        # initial observation
        observation = env.reset()
        env.index = 0
        count +=1
        # if (episode >= 200) and (episode % 200 == 0):
        #     print("episode:",episode)

        print("episode start")
        env.last_reward = 0
        while True:
            # RL choose action based on observation
           
            action = RL.choose_action(observation)
            # print("action",action)

            # RL take action and get next observation and reward
            
            observation_, reward, done = env.step(action)

            reward /= 100
            reward = reward - env.last_reward
          
            if env.index == 0:
              reward = 0
            #print("reward",reward)

            RL.store_transition(observation, action, reward, observation_)
           

            if (step >= 10) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            env.last_reward = reward

            # break while loop when end of this episode
            if done:
                env.reward_his.append(reward)
                reward_total_ql +=  reward
                print("reward_total",reward_total_ql)
                print("episode end")
                break
            env.index +=1
            step += 1
        if RL.memory_counter >= 20000000:
            break
            
    return (reward_total_ql /count)
    
    
    
    
if __name__ == "__main__":
    
    print("DQN")
    start =time.time()
    env = LAA()
    RL = DuelingDQN(15, 300, memory_size=2000,e_greedy_increment=0.00001, sess=None, dueling=True, output_graph=True)
    #RL = DoubleDQN(15, 300, memory_size=2000,e_greedy_increment=0.00001, double_q=True, sess=None, output_graph=True)
    #RL = DeepQNetwork(15, 300,memory_size=2000,output_graph=True)

 
    for i in range(1000):
        
        
        env.SBSs = 5
        env.WIFIAPs= 1
        env.SBSUEs = 20
        env.Bands = 15
        
        if RL.memory_counter >= 20000000:
            break

        for j in range(1):
            
            env.WIFIAPs= 1
            number_of_WIFIAP = []
            ql_total_reward_list = []

            for i in range(5):
            
                # if i >0 :
                #     if env.WIFIAPs == 1:
                #         env.WIFIAPs = i * 5
                #     else:
                #         env.WIFIAPs += 5
            #     number_of_WIFIAP.append(env.WIFIAPs)
            #     print("WIFIAPs:",env.WIFIAPs)

                env.setup_env()

                average = run_LAA()
                ql_total_reward_list.append(average)
            
            print(ql_total_reward_list)
            
            env.SBSs += 5
            
        end = time.time()
        print("執行時間:",end-start)
    
#------------------------------------------------------------------------------------------#




#Test
#------------------------------------------------------------------------------------------#
def run_LAA():

    count = 0
    step = 0
    reward_total_ql_0 = 0
    reward_total_ql_1 = 0
    reward_total_ql_2 = 0
    reward_total_random = 0
    reward_total_op = 0
    for episode in range(100):
      
        # initial observation
        observation = env.reset()
        env.index = 0
        count +=1
        
        observation_0 = observation
        observation_1 = observation
        observation_2 = observation

        #DQN
        RL_0 = DeepQNetwork(15, 300,memory_size=2000,output_graph=True)
        print("DQN")
        while True:

            # RL choose action based on observation
            action = RL_0.choose_action(observation_0)
            #print("action",action)
            
            observation_, reward, done = env.step(action)

            observation_0 = observation_
            print("reward:",reward)
            print("\n")
            # break while loop when end of this episode
            if done:
                #env.reward_his.append(reward)
                reward_total_ql_0 +=  reward
                # print("reward",reward)
                break
            env.index +=1
            step += 1
        
        env.index = 0
        
        tf.reset_default_graph()
        for num in range(env.SBSs*env.SBSUEs):
          env.SBSUE_LOC[num][2] = -1
        

        #DoubleQN
        RL_1 = DoubleDQN(15, 300, memory_size=2000,e_greedy_increment=0.00001, double_q=True, sess=None, output_graph=True)
        print("DoubleDQN")
        while True:
            # RL choose action based on observation

            action = RL_1.choose_action(observation_1)
            #print("action",action)
            
            observation_, reward, done = env.step(action)

            observation_1 = observation_
            print("reward:",reward)
            print("\n")
            # break while loop when end of this episode
            if done:
                #env.reward_his.append(reward)
                reward_total_ql_1 +=  reward
                # print("reward",reward)
                # print("episode end")
                break
            env.index +=1

        env.index = 0
        tf.reset_default_graph()
        for num in range(env.SBSs*env.SBSUEs):
          env.SBSUE_LOC[num][2] = -1

        #DuelingDQN
        RL_2 = DuelingDQN(15, 300, memory_size=1000,e_greedy_increment=0.00001, sess=None, dueling=True, output_graph=True)
        print("DuelingDQN")
        while True:

            # RL choose action based on observation
            action = RL_2.choose_action(observation_2)
            #print("action",action)
            
            observation_, reward, done = env.step(action)

            observation_2 = observation_
            print("reward:",reward)
            print("\n")
            # break while loop when end of this episode
            if done:
                #env.reward_his.append(reward)
                reward_total_ql_2 +=  reward
                # print("reward",reward)
                # print("episode end")
                break
            env.index +=1
            tf.reset_default_graph()
            

        for index in range(env.SBSs*env.SBSUEs):
            if env.choose_list[index]==-1:
                env.choose_action_OP(index)
        
        reward_op = env.get_throughtput_OP()
        #print("reward_op:",reward_op)
        reward_total_op +=  reward_op
        reward_total_op += env.get_throughtput_OP()
        reward_total_random += env.get_throughtput_Random()

    return ((reward_total_ql_0 /count),(reward_total_ql_1 / count),(reward_total_ql_2 / count),(reward_total_op / count),(reward_total_random / count))
    
    

### WIFIAP
if __name__ == "__main__":
    
    start =time.time()
    env = LAA()
   
    

 #1
    for i in range(1):
        
        env.SBSs = 5
        env.WIFIAPs= 1
        env.SBSUEs = 20
        env.Bands = 15
        
        if (i >= 1000) and (i % 1000 == 0):
            print("episode:",i)

        for j in range(1):
            
            env.WIFIAPs= 1
            number_of_WIFIAP = []
            ql_0_total_reward_list = []
            ql_1_total_reward_list = []
            ql_2_total_reward_list = []
            random_total_reward_list = []
            op_total_reward_list = []
            print("SBSs:",env.SBSs)

            for i in range(5):
            
                if i >0 :
                    if env.WIFIAPs == 1:
                        env.WIFIAPs = i * 5
                    else:
                        env.WIFIAPs += 5
                number_of_WIFIAP.append(env.WIFIAPs)
                print("WIFIAPs:",env.WIFIAPs)

                env.setup_env()

                reward_ql_0,reward_ql_1,reward_ql_2,reward_op,reward_random = run_LAA()
            
                ql_0_total_reward_list.append(reward_ql_0)
                ql_1_total_reward_list.append(reward_ql_1)
                ql_2_total_reward_list.append(reward_ql_2)
              

                random_total_reward_list.append(reward_random)
                op_total_reward_list.append(reward_op)
                # RL.plot_cost()
                #env.plot_reward()
                #print(random_total_reward_list)
                #tf.reset_default_graph()
            
            if j==0:
                number_of_WIFIAP_0 = number_of_WIFIAP
                random_total_reward_list_0 = random_total_reward_list
                op_total_reward_list_0 = op_total_reward_list
                ql_0_total_reward_list_0 = ql_0_total_reward_list
                ql_1_total_reward_list_0 = ql_1_total_reward_list
                ql_2_total_reward_list_0 = ql_2_total_reward_list
                
        #     if j==1:
        #         number_of_WIFIAP_1 = number_of_WIFIAP
        #         random_total_reward_list_1 = random_total_reward_list
        #         op_total_reward_list_1 = op_total_reward_list
            
        #     if j==2:
        #         number_of_WIFIAP_2 = number_of_WIFIAP
        #         random_total_reward_list_2 = random_total_reward_list
        #         op_total_reward_list_2 = op_total_reward_list

        #     if j==3:
        #         number_of_WIFIAP_3 = number_of_WIFIAP
        #         random_total_reward_list_3 = random_total_reward_list
        #         op_total_reward_list_3 = op_total_reward_list
                
            
            env.SBSs += 5
            
        # random_line0, = plt.plot(number_of_WIFIAP_0, random_total_reward_list_0,label='Random.SBSs=5',color='b',marker='^',linestyle=':')
        # random_line1, = plt.plot(number_of_WIFIAP_1, random_total_reward_list_1,label='Random.SBSs=10',color='g',marker='v',linestyle=':')
        # random_line2, = plt.plot(number_of_WIFIAP_2, random_total_reward_list_2,label='Random.SBSs=15',color='r',marker='s',linestyle=':')
        # random_line3, = plt.plot(number_of_WIFIAP_3, random_total_reward_list_3,label='Random.SBSs=20',color='c',marker='o',linestyle=':')
        # op_line0, = plt.plot(number_of_WIFIAP_0, op_total_reward_list_0,label='Optimize.SBSs=5',color='b',marker='^',linestyle='-.')
        # op_line1, = plt.plot(number_of_WIFIAP_1, op_total_reward_list_1,label='Optimize.SBSs=10',color='g',marker='v',linestyle='-.')
        # op_line2, = plt.plot(number_of_WIFIAP_2, op_total_reward_list_2,label='Optimize.SBSs=15',color='r',marker='s',linestyle='-.')
        # op_line3, = plt.plot(number_of_WIFIAP_3, op_total_reward_list_3,label='Optimize.SBSs=20',color='c',marker='o',linestyle='-.')

        ql_0_line0, = plt.plot(number_of_WIFIAP_0, ql_0_total_reward_list_0,label='DQN',color='b',marker='^',linestyle='-')
        ql_1_line0, = plt.plot(number_of_WIFIAP_0, ql_1_total_reward_list_0,label='DoubleDQN',color='g',marker='^',linestyle='-.')
        ql_2_line0, = plt.plot(number_of_WIFIAP_0, ql_2_total_reward_list_0,label='DuelingDQN',color='r',marker='^',linestyle='--')
        random_line0, = plt.plot(number_of_WIFIAP_0, random_total_reward_list_0,label='Random',color='m',marker='^',linestyle=':')
        op_line0, = plt.plot(number_of_WIFIAP_0, op_total_reward_list_0,label='OptimizeSOL',color='r',marker='^',linestyle=':')
        plt.legend(handles = [ql_0_line0, ql_1_line0,ql_2_line0,op_line0,random_line0,],loc='best')
        
        # plt.legend(handles = [random_line0, random_line1, random_line2, random_line3, op_line0, op_line1, op_line2, op_line3], loc='best')
        plt.ylabel('Sum rate per SBS (Mbits/s)')
        plt.xlabel('The number of WiFi APs (I)')
        plt.grid()
        plt.savefig('./WiFi.svg')
        plt.savefig('./WiFi.png')
        plt.show()
       
        end = time.time()
        print("執行時間:",end-start)


tf.reset_default_graph()


### SBSUE
if __name__ == "__main__":

    start =time.time()
    env = LAA()
    
    env.SBSs = 5
    env.WIFIAPs= 20
    env.SBSUEs = 1
    env.Bands = 15
    
    

    for j in range(1):
        
        env.SBSUEs= 1
        number_of_SBSUE = []
        ql_0_total_reward_list = []
        ql_1_total_reward_list = []
        ql_2_total_reward_list = []
        op_total_reward_list = []
        random_total_reward_list = []
        print("SBSs:",env.SBSs)

        for i in range(5):

            if i >0 :
                if env.SBSUEs == 1:
                    env.SBSUEs = i * 5
                else:
                    env.SBSUEs += 5
            number_of_SBSUE.append(env.SBSUEs)
            print("SBSUEs:",env.SBSUEs)

            env.setup_env()

            reward_ql_0,reward_ql_1,reward_ql_2,reward_op,reward_random = run_LAA()
            
            ql_0_total_reward_list.append(reward_ql_0)
            ql_1_total_reward_list.append(reward_ql_1)
            ql_2_total_reward_list.append(reward_ql_2)
            
            op_total_reward_list.append(reward_op)
            random_total_reward_list.append(reward_random)
        
        if j==0:
            number_of_SBSUE_0 = number_of_SBSUE
            random_total_reward_list_0 = random_total_reward_list
            op_total_reward_list_0 = op_total_reward_list
            ql_0_total_reward_list_0 = ql_0_total_reward_list
            ql_1_total_reward_list_0 = ql_1_total_reward_list
            ql_2_total_reward_list_0 = ql_2_total_reward_list
        
        env.SBSs += 5
    
    ql_0_line0, = plt.plot(number_of_SBSUE_0, ql_0_total_reward_list_0,label='DQN',color='b',marker='^',linestyle='-')
    ql_1_line0, = plt.plot(number_of_SBSUE_0, ql_1_total_reward_list_0,label='DoubleDQN',color='g',marker='^',linestyle='-.')
    ql_2_line0, = plt.plot(number_of_SBSUE_0, ql_2_total_reward_list_0,label='DuelingDQN',color='r',marker='^',linestyle='--')
    random_line0, = plt.plot(number_of_WIFIAP_0, random_total_reward_list_0,label='Random',color='m',marker='^',linestyle=':')
    op_line0, = plt.plot(number_of_SBSUE_0, op_total_reward_list_0,label='OptimizeSOL',color='r',marker='^',linestyle=':')
    plt.legend(handles = [ql_0_line0, ql_1_line0,ql_2_line0,op_line0,random_line0],loc='best')
        
    plt.ylabel('Sum rate per SBS (Mbits/s)')
    plt.xlabel('The number of LAA UEs (K)')
    plt.grid()
    plt.savefig('./LAA.svg')
    plt.savefig('./LAA.png')
    plt.show()

    end = time.time()
    print("執行時間:",end-start)

