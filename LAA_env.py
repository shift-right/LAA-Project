import random
import math
import numpy as np
import matplotlib.pyplot as plt

class LAA():
    def __init__(self):
        self.action_space = []  #['0','1','2','3','4'] depend on band
        self.n_actions = 0 # number of band
        self.n_features = 0  # (SBSs+WIFIAPs+SBSUEs)*3
        self.SBSs = 0
        self.WIFIAPs = 0    #per SBS
        self.SBSUEs = 0     #per SBS
        #self.WIFIAPUEs = 0
        self.Bands = 0
        self.SBS_LOC= []
        self.WIFIAP_LOC= []
        self.MBS_COR = [[0,0]]
        self.SBSUE_LOC = []
        #self.WIFIAPUE_LOC = []
        self.MBS_radius = 500
        self.SBS_radius = 60
        self.WIFIAP_radius = 50
        self.ALL = []
        self.reward_his = []
        self.index = 0
        self.WIFIAPBAND = []
        self.d_SBSUE_to_SBS = []
        self.d_SBSUE_to_WIFIAP = []
        self.d_SBSUE_to_SBSUE = []
        self.fading_gain_SBS_SBSUE = []
        self.fading_gain_WIFIAP_SBSUE = []
        self.fading_gain_SBSUE_SBSUE = []
        self.Total_fading_gain_SBS_SBSUE = []
        self.choose_list = []
        self.last_reward = 0
        


   
    def get_random_point(self,times:'int',radius:'float',COR=[],list=[]):
        x_center = COR[0][0]
        y_center = COR[0][1]
        x_max = x_center + radius
        y_max = y_center + radius
        x_min = x_center - radius
        y_min = y_center - radius
        #print(times)
        for count in range(times):
            res_x = random.uniform(x_min, x_max)
            res_y = random.uniform(y_min, y_max)
            dis = (res_x - x_center)**2 + (res_y - y_center)**2
            while(dis>radius**2): 
                res_x = random.uniform(x_min, x_max)
                res_y = random.uniform(y_min, y_max)
                dis = (res_x - x_center)**2 + (res_y - y_center)**2
            list.append([res_x,res_y,-1])
    def get_random_SBS_LOC(self):
        self.get_random_point(self.SBSs,self.MBS_radius,self.MBS_COR,self.SBS_LOC) 
        

    def get_random_SBSUE_LOC(self):
        for i in range(self.SBSs):
            #print("i_SBSUE_LOC:",i)
            S_COR = [[self.SBS_LOC[i][0],self.SBS_LOC[i][1]]]
            self.get_random_point(self.SBSUEs,self.SBS_radius,S_COR,self.SBSUE_LOC)
        
        #print("self.SBSUE_LOC",self.SBSUE_LOC)

    def get_random_WIFIAP_LOC(self):
        for i in range(self.SBSs):
            #print("i_WIFIAP_LOC:",i)
            S_COR = [[self.SBS_LOC[i][0],self.SBS_LOC[i][1]]]
            #print("S_COR:",S_COR)
            self.get_random_point(self.WIFIAPs,self.SBS_radius,S_COR,self.WIFIAP_LOC)

        for count in range(self.SBSs * self.WIFIAPs):
            CHOOSE_BAND = random.randint(0,self.Bands-1)
            self.WIFIAP_LOC[count][2] = CHOOSE_BAND
            self.WIFIAPBAND.append(CHOOSE_BAND)
        #print("self.WIFIAP_LOC",self.WIFIAP_LOC)

    # def get_random_WIFIAPUE_LOC(self):
    #     for count in range(self.WIFIAPUEs):
    #         CHOOSE_AP = random.randint(0,self.WIFIAPs-1) #WIFIAPåŸºç«™3å€‹
    #         W_COR = [[self.WIFIAP_LOC[CHOOSE_AP][0],self.WIFIAP_LOC[CHOOSE_AP][1]]]
    #         self.get_random_point(1,self.WIFIAP_radius,W_COR,self.WIFIAPUE_LOC)
    #         self.WIFIAPUE_LOC[count][2] = self.WIFIAP_LOC[CHOOSE_AP][2]
        
            
        
        
    
    def get_distance_to_other_same(self,UEs:'int',index:'int',LOC=[],DIS=[]):
        for i in range(UEs):
            distance = ((LOC[index][0]-LOC[i][0])**2 + (LOC[index][1]-LOC[i][1])**2)**(1/2)
            DIS.append(distance)
    
    def get_distance_to_other_different(self,UEs:'int',index:'int',LOC1=[],LOC2=[],DIS=[]):
        for i in range(UEs):
            distance = ((LOC1[index][0]-LOC2[i][0])**2 + (LOC1[index][1]-LOC2[i][1])**2)**(1/2)
            # print("\n")
            # print("LOC1[index][0],LOC1[index][1]")
            # print(LOC1[index][0],LOC1[index][1])
            # print("LOC2[i][0],LOC2[i][1]")
            # print(LOC2[i][0],LOC2[i][1])
            # print("\n")
            DIS.append(distance)

    def get_gain(self,UEs:'int',d=[]):
        gain=[]
        #print("get_gain UEs:",UEs)
        for i in range(UEs):
            if d[i]==0:
                g2 = 0
            elif d[i]<=300:
                g2_db = -22 * math.log(d[i],10)-28-13.9794
                g2 = 10 ** (g2_db/10)
            elif (d[i]>300) and (d[i]<5000):
                g2_db = -40 * math.log(d[i],10) -7.8 + 17.1764 - 5.4185 -1.3979
                g2 = 10 ** (g2_db/10)
            else:
                g2=0
            gain.append(g2)
        
        gain = np.array(gain)
        #print("gain:",gain)
        return gain

    def get_beta(self,UEs:'int'):
        beta1_db_shadow = []
        for i in range(UEs):
            beta1_db_shadow.append(3*random.random())
        beta1_db_shadow=np.array(beta1_db_shadow)
        beta1_db = 10**(beta1_db_shadow/10)
        beta = beta1_db
        #print("beta:",beta)
        return beta
    
    def get_Rayleigh_fading_gain(self,UEs:'int'):
        Rayleigh_fading_gain = []
        for i in range(UEs):
            Rayleigh_fading_gain.append(np.random.rayleigh(0.0302))
        Rayleigh_fading_gain = np.array(Rayleigh_fading_gain)
        #print("Rayleigh_fading_gain:",Rayleigh_fading_gain)
        return Rayleigh_fading_gain

    def get_fading_gain(self,UEs:'int',dis=[]):
        gain = self.get_gain(UEs,dis)
        beta = self.get_beta(UEs)
        Rayleigh_fading_gain = self.get_Rayleigh_fading_gain(UEs)
        return gain*beta*Rayleigh_fading_gain

    def get_distance_fading_gain(self):
        Total_fading_gain_SBS_SBSUE = []
        #------------------------------------------------------------------------------------------------------------
        #SBS_to_SBSUE   
        num_t1 = self.SBSs*self.SBSUEs
        for t1_1 in range(num_t1):
            self.get_distance_to_other_different(self.SBSs,t1_1,self.SBSUE_LOC,self.SBS_LOC,self.d_SBSUE_to_SBS)
            #print("self.d_SBSUE_to_SBS",self.d_SBSUE_to_SBS)

        self.fading_gain_SBS_SBSUE = self.get_fading_gain(self.SBSs*num_t1,self.d_SBSUE_to_SBS)
        #print("self.fading_gain_SBS_SBSUE",self.fading_gain_SBS_SBSUE)
        for t1_2 in range(num_t1):
            count_t1 = 0
            for t1_3 in range(self.SBSs):
                #print("t1_3:",t1_3)
                #print("fading_gain_SBS_SBSUE[num_t1*t1_2+t1_3]",fading_gain_SBS_SBSUE[num_t1*t1_2+t1_3])
                count_t1 += self.fading_gain_SBS_SBSUE[t1_2*self.SBSs+t1_3]
                #print("count_t1",count_t1)
            Total_fading_gain_SBS_SBSUE.append(count_t1)
        self.Total_fading_gain_SBS_SBSUE = np.array(Total_fading_gain_SBS_SBSUE)  
        #print("self.Total_fading_gain_SBS_SBSUE",self.Total_fading_gain_SBS_SBSUE)
        #------------------------------------------------------------------------------------------------------------
        #WIFIAP_to_SBSUE        ###check_table
        num_t2_1 = self.SBSs*self.SBSUEs
        num_t2_2 = self.SBSs*self.WIFIAPs
        
        for t2_1 in range(num_t2_1):
            self.get_distance_to_other_different(num_t2_2,t2_1,self.SBSUE_LOC,self.WIFIAP_LOC,self.d_SBSUE_to_WIFIAP)
        #print("self.d_SBSUE_to_WIFIAP",self.d_SBSUE_to_WIFIAP)
       
        self.fading_gain_WIFIAP_SBSUE = self.get_fading_gain(num_t2_1*num_t2_2,self.d_SBSUE_to_WIFIAP)
        #print("self.fading_gain_WIFIAP_SBSUE",self.fading_gain_WIFIAP_SBSUE)
        #print("WIFIAP",self.WIFIAP_LOC)
        #print("SBSUE",self.SBSUE_LOC)
        #------------------------------------------------------------------------------------------------------------
        #SBSUE_to_SBSUE         ###check_table
        num_t3 = self.SBSs*self.SBSUEs
        for t3_1 in range(num_t3):
            self.get_distance_to_other_same(num_t3,t3_1,self.SBSUE_LOC,self.d_SBSUE_to_SBSUE)
        #print("self.d_SBSUE_to_SBSUE",self.d_SBSUE_to_SBSUE)
        
        self.fading_gain_SBSUE_SBSUE = self.get_fading_gain(num_t3*num_t3,self.d_SBSUE_to_SBSUE)
        #print("self.fading_gain_SBSUE_SBSUE",self.fading_gain_SBSUE_SBSUE)
        
  




    def get_throughtput_QL(self):  

        # d_SBS_to_SBSUE = []
        # d_WIFIAP_to_SBSUE = []
        # d_SBSUE_to_SBSUE = []
        Total_fading_gain_WIFIAP_SBSUE = []
        Total_fading_gain_SBSUE_SBSUE = []
        Num = 0
        Deno = 0
        Throughtput = []
        Total_Throughtput = 0
        Average_Throughtput = 0
        


        
        # #WIFIAP_to_SBSUE        ###check_table
        num_t2_1 = self.SBSs*self.SBSUEs
        num_t2_2 = self.SBSs*self.WIFIAPs
      
        for t2_1 in range(num_t2_1):
            count_t2 = 0
            for t2_2 in range(num_t2_2):
                if self.SBSUE_LOC[t2_1][2] ==  self.WIFIAP_LOC[t2_2][2] != -1 :
                    count_t2 += self.fading_gain_WIFIAP_SBSUE[t2_1*num_t2_2+ t2_2]
            Total_fading_gain_WIFIAP_SBSUE.append(count_t2)
        Total_fading_gain_WIFIAP_SBSUE = np.array(Total_fading_gain_WIFIAP_SBSUE)
        #print("Total_fading_gain_WIFIAP_SBSUE:",Total_fading_gain_WIFIAP_SBSUE)

        #------------------------------------------------------------------------------------------------------------
        
        # #SBSUE_to_SBSUE         ###check_table
        num_t3 = self.SBSs*self.SBSUEs
        # for t3_1 in range(num_t3):
        #     self.get_distance_to_other_same(num_t3,t3_1,self.SBSUE_LOC,d_SBSUE_to_SBSUE)
        # #print("d_SBSUE_to_SBSUE",d_SBSUE_to_SBSUE)
        
        # fading_gain_SBSUE_SBSUE = self.get_fading_gain(num_t3*num_t3,d_SBSUE_to_SBSUE)
        # #print("fading_gain_SBSUE_SBSUE",fading_gain_SBSUE_SBSUE)
        
        # #print("SBSUE",self.SBSUE_LOC)
        for t3_1 in range(num_t3):
            count_t3 = 0
            for t3_2 in range(num_t3):
                if self.SBSUE_LOC[t3_1][2] == self.SBSUE_LOC[t3_2][2] != -1:
                    count_t3 += self.fading_gain_SBSUE_SBSUE[t3_1*num_t3 + t3_2]
            Total_fading_gain_SBSUE_SBSUE.append(count_t3)
        Total_fading_gain_SBSUE_SBSUE = np.array(Total_fading_gain_SBSUE_SBSUE)
        #print("Total_fading_gain_SBSUE_SBSUE:",Total_fading_gain_SBSUE_SBSUE)

        # for t3_2 in range(self.SBSs):  
        #     count_t3 = 0
        #     #print("t3_2",t3_2) 
        #     for t3_3 in range(self.SBSUEs):
        #         #print("t3_3",t3_3)
        #         #print("left index:",t3_2*self.SBSUEs+t3_3)
        #         #print("left value:",self.SBSUE_LOC[t3_2*self.SBSUEs+t3_3][2])
        #         for t3_4 in range(num_t3):
        #             #print("right index:",t3_4)
        #             #print("right value:",self.SBSUE_LOC[t3_4][2])
        #             #print("fading gain index:",(t3_2*self.SBSUEs+t3_3)*num_t3+t3_4)
        #             if self.SBSUE_LOC[t3_2*self.SBSUEs+t3_3][2] == self.SBSUE_LOC[t3_4][2] != -1 :
        #                 count_t3 += self.fading_gain_SBSUE_SBSUE[(t3_2*self.SBSUEs+t3_3)*num_t3+t3_4]
        #                 #print("same")
        #                 #print("count_t3",count_t3)
                    
        #     Total_fading_gain_SBSUE_SBSUE.append(count_t3)
        # Total_fading_gain_SBSUE_SBSUE = np.array(Total_fading_gain_SBSUE_SBSUE)
        # #print("Total_fading_gain_SBSUE_SBSUE:",Total_fading_gain_SBSUE_SBSUE)
        #------------------------------------------------------------------------------------------------------------
        #Variable
        P_LAA = (10**(-3))*(10**(24/10))
        Tmax = 10
        Icca = 0.0034
        Ecca = 0.0009
        No = 2 * (10 ** (-13))
        WiFi_Tx_Power = 0.1
        P_max = 0.2512
        #------------------------------------------------------------------------------------------------------------
        Num = P_LAA * self.Total_fading_gain_SBS_SBSUE
        #print("Num",Num)
        
        # ### for training
        # Deno = P_max * Total_fading_gain_SBSUE_SBSUE + No
        # ###
        
        
        Deno = WiFi_Tx_Power * Total_fading_gain_WIFIAP_SBSUE + P_max * Total_fading_gain_SBSUE_SBSUE + No
        #print("Deno",Deno)
        Throughtput = 20 * Tmax * np.log10(1+(Num/Deno))/(Icca+Tmax)
        
        
        #print("Throughtput",Throughtput)
        for i in range(num_t3):
            Total_Throughtput += Throughtput[i]
        #     if Throughtput[i]!=0:
        #         flag=1
        #     else:
        #         flag=0
        #         break
        # if flag==1:
        #     print("Throughtput",Throughtput)
        # print("Total_Throughtput",Total_Throughtput)
        Average_Throughtput = Total_Throughtput/self.SBSs
        #print("Average_Throughtput",Average_Throughtput)
        
        #just for training
        
        
        return Average_Throughtput
        #return Total_Throughtput
        #return Average_Throughtput
        

    def choose_action_OP(self,index:'int'):

        empty_band = self.Bands 
        #print("self.choose_list",self.choose_list)
        CH_list = [i for i in range(self.Bands)] 
        #print("CH_list",CH_list)

        max_dis = 0
        max_dis_index = -1
        for i in range(self.SBSs*self.SBSUEs):
            if self.d_SBSUE_to_SBSUE[index*self.SBSs*self.SBSUEs+i] > max_dis and self.choose_list[i]!= -1:
                max_dis = self.d_SBSUE_to_SBSUE[index*self.SBSs*self.SBSUEs+i]
                max_dis_index = self.choose_list[i]

            if self.d_SBSUE_to_SBSUE[index*self.SBSs*self.SBSUEs+i] < self.SBS_radius:
                try:
                    CH_list.remove(self.choose_list[i])
                except:
                    continue

        empty_band = len(CH_list)
        if empty_band > 0:
            self.choose_list[index] = random.choice(CH_list)
        else:
            self.choose_list[index] = max_dis_index

    def get_throughtput_OP(self):
        Num = 0
        Deno = 0
        Throughtput = []
        Total_Throughtput = 0
        Average_Throughtput = 0
        Total_fading_gain_WIFIAP_SBSUE = []
        Total_fading_gain_SBSUE_SBSUE = []
        #------------------------------------------------------------------------------------------------------------
        # #WIFIAP_to_SBSUE        ###check_table
        num_t2_1 = self.SBSs*self.SBSUEs
        num_t2_2 = self.SBSs*self.WIFIAPs

        for t2_1 in range(num_t2_1):
            count_t2 = 0
            for t2_2 in range(num_t2_2):
                if self.choose_list[t2_1] == self.WIFIAPBAND[t2_2] != -1 :
                    count_t2 += self.fading_gain_WIFIAP_SBSUE[t2_1*num_t2_2+ t2_2]
            Total_fading_gain_WIFIAP_SBSUE.append(count_t2)
        Total_fading_gain_WIFIAP_SBSUE = np.array(Total_fading_gain_WIFIAP_SBSUE)
        #print("Total_fading_gain_WIFIAP_SBSUE:",Total_fading_gain_WIFIAP_SBSUE)
       
        #------------------------------------------------------------------------------------------------------------
        # #SBSUE_to_SBSUE         ###check_table
        num_t3 = self.SBSs*self.SBSUEs
        for t3_1 in range(num_t3):
            count_t3 = 0
            for t3_2 in range(num_t3):
                if self.choose_list[t3_1] == self.choose_list[t3_2] != -1:
                    count_t3 += self.fading_gain_SBSUE_SBSUE[t3_1*num_t3 + t3_2]
            Total_fading_gain_SBSUE_SBSUE.append(count_t3)
        Total_fading_gain_SBSUE_SBSUE = np.array(Total_fading_gain_SBSUE_SBSUE)
        #print("Total_fading_gain_SBSUE_SBSUE:",Total_fading_gain_SBSUE_SBSUE)
       
        #------------------------------------------------------------------------------------------------------------
        #Variable
        P_LAA = (10**(-3))*(10**(24/10))
        Tmax = 10
        Icca = 0.0034
        Ecca = 0.0009
        No = 2 * (10 ** (-13))
        WiFi_Tx_Power = 0.1
        P_max = 0.2512
        #------------------------------------------------------------------------------------------------------------
        Num = P_LAA * self.Total_fading_gain_SBS_SBSUE
        #print("Num",Num)
        Deno = WiFi_Tx_Power * Total_fading_gain_WIFIAP_SBSUE + P_max * Total_fading_gain_SBSUE_SBSUE + No
        #print("Deno",Deno)
        Throughtput = 20 * Tmax * np.log10(1+(Num/Deno))/(Icca+Tmax)
        #print("Throughtput",Throughtput)
        
        for i in range(num_t3):
            Total_Throughtput += Throughtput[i]
        
        # print("Total_Throughtput",Total_Throughtput)
        Average_Throughtput = Total_Throughtput/self.SBSs
        #print("Average_Throughtput",Average_Throughtput)
        #return Total_Throughtput
        return Average_Throughtput

    def get_throughtput_Random(self):
        SBSUE_Band = []
        Num = 0
        Deno = 0
        Throughtput = []
        Total_Throughtput = 0
        Average_Throughtput = 0
        Total_fading_gain_WIFIAP_SBSUE = []
        Total_fading_gain_SBSUE_SBSUE = []
        for i in range(self.SBSs*self.SBSUEs):
            SBSUE_Band.append(random.randint(0,self.Bands))  ########
        #------------------------------------------------------------------------------------------------------------
        # #WIFIAP_to_SBSUE        ###check_table
        num_t2_1 = self.SBSs*self.SBSUEs
        num_t2_2 = self.SBSs*self.WIFIAPs

        for t2_1 in range(num_t2_1):
            count_t2 = 0
            for t2_2 in range(num_t2_2):
                if SBSUE_Band[t2_1] == self.WIFIAPBAND[t2_2] != -1 :
                    count_t2 += self.fading_gain_WIFIAP_SBSUE[t2_1*num_t2_2+ t2_2]
            Total_fading_gain_WIFIAP_SBSUE.append(count_t2)
        Total_fading_gain_WIFIAP_SBSUE = np.array(Total_fading_gain_WIFIAP_SBSUE)
        #print("Total_fading_gain_WIFIAP_SBSUE:",Total_fading_gain_WIFIAP_SBSUE)

        #------------------------------------------------------------------------------------------------------------
        # #SBSUE_to_SBSUE         ###check_table
        num_t3 = self.SBSs*self.SBSUEs
        for t3_1 in range(num_t3):
            count_t3 = 0
            for t3_2 in range(num_t3):
                if SBSUE_Band[t3_1] == SBSUE_Band[t3_2] != -1 :
                    count_t3 += self.fading_gain_SBSUE_SBSUE[t3_1*num_t3 + t3_2]
            Total_fading_gain_SBSUE_SBSUE.append(count_t3)
        Total_fading_gain_SBSUE_SBSUE = np.array(Total_fading_gain_SBSUE_SBSUE)
        #print("Total_fading_gain_SBSUE_SBSUE:",Total_fading_gain_SBSUE_SBSUE)

        #------------------------------------------------------------------------------------------------------------
        #Variable
        P_LAA = (10**(-3))*(10**(24/10))
        Tmax = 10
        Icca = 0.0034
        Ecca = 0.0009
        No = 2 * (10 ** (-13))
        WiFi_Tx_Power = 0.1
        P_max = 0.2512
        #------------------------------------------------------------------------------------------------------------
        Num = P_LAA * self.Total_fading_gain_SBS_SBSUE
        #print("Num",Num)
        Deno = WiFi_Tx_Power * Total_fading_gain_WIFIAP_SBSUE + P_max * Total_fading_gain_SBSUE_SBSUE + No
        #print("Deno",Deno)
        Throughtput = 20 * Tmax * np.log10(1+(Num/Deno))/(Icca+Tmax)
        #print("Throughtput",Throughtput)
        
        for i in range(num_t3):
            Total_Throughtput += Throughtput[i]
       
        # print("Total_Throughtput",Total_Throughtput)
        Average_Throughtput = Total_Throughtput/self.SBSs
        #print("Average_Throughtput",Average_Throughtput)
        #return Total_Throughtput
        return Average_Throughtput


    def get_ALL(self):
        #self.ALL = self.SBS_LOC + self.WIFIAP_LOC + self.SBSUE_LOC
        self.ALL = self.SBSUE_LOC

 
    def step(self, action):

        s = self.ALL
        #count = 2+(self.SBSs + self.SBSs*self.WIFIAPs + self.index) * 3
        count = 2+(self.index) * 3
        s[count] = action
        self.SBSUE_LOC[self.index][2] = action
        reward = self.get_throughtput_QL()


        
        s_ = np.array(s)
        if self.index == self.SBSs*self.SBSUEs-1:
            done = True
        else:
            done = False
        # return s_, reward+Bonus_Reward, done
        return s_, reward, done

    def reduce_dim(self):
        lis = str(self.ALL)
        lis = lis.replace('[','')
        lis = lis.replace(']','')
        self.ALL = list(eval(lis))
        for i in range(300-len(self.ALL)):
            self.ALL.append(-1)

    def setup_env(self):
        for i in range(self.Bands):
            self.action_space.append(i)
        self.n_actions = len(self.action_space)
        #self.n_features = 3 * (self.SBSs*(self.WIFIAPs+self.SBSUEs) + self.SBSs)
        self.n_features = 3 * (self.SBSs*self.SBSUEs)
        
        

    def reset(self):
        self.SBS_LOC= []
        self.WIFIAP_LOC= []
        self.SBSUE_LOC = []
        self.WIFIAPUE_LOC = []
        self.ALL = []
        self.d_SBSUE_to_SBS = []
        self.d_SBSUE_to_WIFIAP = []
        self.d_SBSUE_to_SBSUE = []
        self.fading_gain_SBS_SBSUE = []
        self.fading_gain_WIFIAP_SBSUE = []
        self.fading_gain_SBSUE_SBSUE = []
        self.Total_fading_gain_SBS_SBSUE = []
        self.choose_list = [-1 for i in range(self.SBSs*self.SBSUEs)]
        self.WIFIAPBAND = []
        self.get_random_SBS_LOC()
        self.get_random_WIFIAP_LOC()
        self.get_random_SBSUE_LOC()
        #self.get_random_WIFIAPUE_LOC()
        self.get_ALL()
        self.reduce_dim()
        self.get_distance_fading_gain()
        #self.get_throughtput_OP()
    
        return np.array(self.ALL)

    
    def plot_reward(self):
        plt.plot(np.arange(len(self.reward_his)), self.reward_his)
        plt.ylabel('Reward')
        plt.xlabel('training steps')
        plt.show()







