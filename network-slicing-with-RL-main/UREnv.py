from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt

# parameters of celluar environment
ser_cat_vec = ['volte', 'embb_general', 'urllc']
band_whole_no = 10 * 10**6
band_per = 1 * 10**6
qoe_weight = [1, 4, 6]
se_weight = 0.01
dl_mimo = 64
learning_window = 2000
rewards = []
observations = []
action0 = []
action1 = []
action2 = []
SE = []
Util= []
QoE0 = []
QoE1 = []
QoE2 = []
total_timesteps=2000

class MyEnv(Env):
    def __init__(self,
        BS_pos = np.array([0,0]), #기지국 위치
        BS_radius = 40, #기지국 신호 범위(셀 반지름)
        BS_tx_power = 16, #unit is dBW, 46dBm, 기지국 전송 파워
        UE_max_no = 100, #최대 사용자 수
        Queue_max = 5,
        noise_PSD = -204, # -174 dbm/Hz, 노이즈 파워 스펙트럼 밀도
        chan_mod = '36814',
        carrier_freq = 2 * 10 ** 9, #2 GHz
        time_subframe = 0.5 * 10 ** (-3), # by LTE, 0.5 ms
        ser_cat = ['volte','embb_general','urllc'], #서비스 종류
        band_whole = 10 * 10 ** 6, # 10MHz(전체 대역폭)
        schedu_method = 'round_robin',
        ser_prob = np.array([6,6,1], dtype=np.float32), #사용자 서비스 비율
        dl_mimo = 64, #MIMO 안테나 수
        rx_gain = 20, #dB
        learning_windows = 60000, #학습 윈도우 크기(시간 간격 기준)
        t=0,
        ):
        self.BS_tx_power = BS_tx_power
        self.BS_radius = BS_radius
        self.band_whole = band_whole
        self.chan_mod = chan_mod
        self.carrier_freq = carrier_freq
        self.time_subframe = round(time_subframe,4)
        self.noise_PSD = noise_PSD
        self.sys_clock = 0
        self.schedu_method = schedu_method 
        self.dl_mimo = dl_mimo
        self.UE_rx_gain = rx_gain
        self.UE_max_no = UE_max_no
        self.UE_buffer = np.zeros([Queue_max,UE_max_no])
        self.UE_buffer_backup = np.zeros([Queue_max,UE_max_no])
        self.UE_latency = np.zeros([Queue_max,UE_max_no])
        self.UE_readtime = np.zeros(UE_max_no)
        self.UE_band = np.zeros(UE_max_no)
        UE_pos = np.random.uniform(-self.BS_radius, self.BS_radius, [self.UE_max_no,2])
        dis = np.sqrt(np.sum((BS_pos - UE_pos) **2 , axis = 1)) / 1000 # unit changes to km
        self.path_loss = 145.4 + 37.5 * np.log10(dis).reshape(-1,1)
        self.learning_windows = round(learning_windows*self.time_subframe,4)
        self.ser_cat = ser_cat #서비스 종류(volte, embb, urllc)
        self.t=t
        if len(self.ser_cat) > 1:
            self.band_ser_cat = np.zeros(len(ser_cat)) #각 서비스가 얼마나 주파수대역폭을 가질지를 기록할 배열을 초기에 0으로 세팅
            if len(ser_prob) == len(self.ser_cat):
                self.ser_prob = ser_prob / np.sum(ser_prob) 
            else:
                self.ser_prob = np.ones(len(ser_cat)) / len(ser_cat)
        else:
            self.ser_prob = np.array([1])
            self.band_ser_cat = self.band_whole

        self.UE_cat = np.random.choice(self.ser_cat, self.UE_max_no, p=self.ser_prob) #TBD
        self.tx_pkt_no = np.zeros(len(self.ser_cat)) #전송한 패킷 수 초기화화


        #observation and action space
        # print("in init")
        self.action_space = Discrete(35,)
        print(self.action_space)
        self.observation_space = Box(low=0, high=1000000, shape=(len(ser_cat),), dtype=np.int32)
        
    


    def step(self, action):
        print("##################################")
        
        #this_action = set(self._action_to_direction[action])
        hold_action = get_act(action) #에이전트가 선택한 행동번호 action을 실제 자원 분배 리스트로 바꿈
        #ex) action=2 -> hold_action=[1,3,6](대역폭 비율)
        this_action=np.multiply(hold_action,10**6) #[MHz]단위를 [Hz]단위로 변경
        self.countReset() #전송 성공한 패킷 수, 총 보낸 패킷 수, SE 기록 등을 초기화
        self.activity() #각 사용자(UE)에게 패킷을 생성하고 버퍼에 저장
        # Select and perform an action
        #observations.append(observation.tolist())
        print("actionnnnnnnnnnnn :", this_action)
        action0.append(this_action[0])
        action1.append(this_action[1])
        action2.append(this_action[2])
        #각 서비스에 할당된 대역폭을 따로 기록
        
        # print("teeeeeeeeeeest", action0)
        self.band_ser_cat = this_action #세가지 서비스에 각각 얼마의 대역폭을 할당할지 자징
        #prev_observation = observation

        for i in itertools.count(): #학습 윈도우가 끝날 때까지 아래 작업 반복
            self.scheduling() #할당된 대역폭을 사용자들에게 분배
            self.provisioning() #사용자들이 받은 리소스로 버퍼에 쌓인 데이터 전송
            if i == (learning_window - 1):
                break
            else:
                self.bufferClear() #다 전송된 패킷의 버퍼 정리리
                self.activity() #새로 도착한 데이터를 다시 버퍼에 채움
        
        qoe, se = self.get_reward() #각 서비스의 QoE, SE 계산
        threshold = 6 + 4 * self.t / (total_timesteps/1.5) #시간이 지날수록 보상 기분이 올라가도록 임계값 설정
        # print("tresh",threshold)

        utility, reward = calc_reward(qoe, se, threshold)
        print("reward",reward)
        Util.append(utility)
        QoE0.append(qoe[0])
        QoE1.append(qoe[1])
        QoE2.append(qoe[2])
        SE.append(se[0])
        rewards.append(reward)
        # kpis.append(kpi)
        # print("num of packets: ",self.tx_pkt_no)
        observation = state_update(self.tx_pkt_no, self.ser_cat) #다음 상태 계산(에이전트가 다음 행동을 할 때 입력값이 됨)
        # print("z score of packets: ",observation)
        # observation = torch.from_numpy(observation).unsqueeze(0).to(torch.float)
        done=False
        if self.t == total_timesteps: #학습 끝났으면 그 동안의 행동,QoE,SE,유틸리티를 모두 그래프로 시각화화
            plt.figure(3)
            plt.plot( np.array(action0),'s',markersize=1, c='r', label='volte')
            
            plt.legend(loc='best')
            plt.ylabel('actions')
            plt.xlabel('training steps')
            plt.grid()

            plt.figure(4)
            
            plt.plot(np.array(action1),'s',markersize=1, c='b', label='embb_general')

            plt.legend(loc='best')
            plt.ylabel('actions')
            plt.xlabel('training steps')
            plt.grid()

            plt.figure(5)
            
            plt.plot(np.array(action2),'s',markersize=1, c='g', label='urllc')
            plt.legend(loc='best')
            plt.ylabel('actions')
            plt.xlabel('training steps')
            plt.grid()

            plt.figure(6)
            plt.plot( np.array(QoE0),'s',markersize=1, c='r', label='volte')
            
            plt.legend(loc='best')
            plt.ylabel('SSR')
            plt.xlabel('training steps')
            plt.grid()

            plt.figure(7)
            
            plt.plot(np.array(QoE1),'s',markersize=1, c='b', label='embb_general')

            plt.legend(loc='best')
            plt.ylabel('SSR')
            plt.xlabel('training steps')
            plt.grid()

            plt.figure(8)
            
            plt.plot(np.array(QoE2),'s',markersize=1, c='g', label='urllc')
            plt.legend(loc='best')
            plt.ylabel('SSR')
            plt.xlabel('training steps')
            plt.grid()

            plt.figure(9)

            plt.plot(np.array(SE),'s',markersize=1, c='r')
            plt.legend(loc='best')
            plt.ylabel('SE')
            plt.xlabel('training steps')
            plt.grid()

            plt.figure(10)

            plt.plot(np.array(Util),'s',markersize=1, c='b')
            plt.legend(loc='best')
            plt.ylabel('Utility')
            plt.xlabel('training steps')
            plt.grid()
            
            plt.show()
            Util.clear()
            SE.clear()
            QoE0.clear()
            QoE1.clear()
            QoE2.clear()
            action0.clear()
            action1.clear()
            action2.clear()

            done = True
        info={}
        self.t += 1 #시간(step) 1 증가
        if done==False:
            print("t :",self.t)

        print("##################################")
        return observation, reward, done, info #에이전트가 사용할 결과값 반환환

    def render(self, mode):
        self.mode =mode
        pass

    def reset(self): #환경을 리셋할 때 실행되는 함수
      self.t=0 #타임스텝을 0으로 초기화
      self.bufferClear() #사용자들의 버퍼랑 지연시간(latency) 관련 데이터를 모두 초기화
      self.countReset()  #전송성공패킷수&전체패킷수&시스템 throughput 모두 초기화
    #   self.activity()
      return 
            ser_cat = len(self.ser_cat)
            band_ser_cat = self.band_ser_cat #각 서비스에 할당된 전체 대역폭
            if (self.sys_clock * 10000) % (self.learning_windows * 10000) == (self.time_subframe * 10000):
                self.ser_schedu_ind =  [0] * ser_cat
            #print("self.UE_buffer",self.UE_buffer)    
            for i in range(ser_cat): 
                UE_index = np.where((self.UE_buffer[0,:]!=0) & (self.UE_cat == self.ser_cat[i]))[0] #현재 데이터버퍼가 비어있지 않으며, 해당 서비스 카테고리에 속한 사용자들
                UE_Active_No = len(UE_index) #전송할 데이터를 가진 사용자 수(활성 사용자 수)
                if UE_Active_No != 0:
                    RB_No = band_ser_cat[i] // (180 * 10**3)
                    RB_round = RB_No // UE_Active_No
                    self.UE_band[UE_index] += 180 * 10**3 * RB_round
                    # print("UEEEEEEE",RB_No)
                    RB_rem_no = int(RB_No - RB_round * UE_Active_No) #나누고 남은 RB(Resource Block) 개수 계산
                    left_no
        self.channel_model()
        rx_power = 10 ** ((self.BS_tx_power - self.chan_loss + self.UE_rx_gain)/10)
        rx_power = rx_power.reshape(1,-1)[0]
        rate = np.zeros(self.UE_max_no)
        # print("UE_band:  ",self.UE_band)
        rate[UE_index] = self.UE_band[UE_index] * np.log10(1 + rx_power[UE_index] / ( 10 **(self.noise_PSD /10) * self.UE_band[UE_index] )) * self.dl_mimo
        buffer = np.sum(self.UE_buffer,axis=0)
        UE_index_b = np.where(buffer != 0) 

        for ue_id in UE_index_b[0]:
            self.UE_latency[:,ue_id]=latencyUpdate(self.UE_latency[:,ue_id],self.UE_buffer[:,ue_id],self.time_subframe)
        
        for ue_id in UE_index[0]: 
            self.UE_buffer[:,ue_id]=bufferUpdate(self.UE_buffer[:,ue_id],rate[ue_id],self.time_subframe)  

        self.store_reward(rate)

        self.bufferClear()
        
 def provisioning(self):
        UE_index = np.where(self.UE_band != 0) 
        self.channel_model()
        rx_power = 10 ** ((self.BS_tx_power - self.chan_loss + self.UE_rx_gain)/10)
        rx_power = rx_power.reshape(1,-1)[0]
        rate = np.zeros(self.UE_max_no)
        # print("UE_band:  ",self.UE_band)
        rate[UE_index] = self.UE_band[UE_index] * np.log10(1 + rx_power[UE_index] / ( 10 **(self.noise_PSD /10) * self.UE_band[UE_index] )) * self.dl_mimo
        buffer = np.sum(self.UE_buffer,axis=0)
        UE_index_b = np.where(buffer != 0) 

        for ue_id in UE_index_b[0]:
            self.UE_latency[:,ue_id]=latencyUpdate(self.UE_latency[:,ue_id],self.UE_buffer[:,ue_id],self.time_subframe)
        
        for ue_id in UE_index[0]: 
            self.UE_buffer[:,ue_id]=bufferUpdate(self.UE_buffer[:,ue_id],rate[ue_id],self.time_subframe)  

        self.store_reward(rate)

        self.bufferClear()
    
    def activity(self): #https://www.ngmn.org/fileadmin/user_upload/NGMN_Radio_Access_Performance_Evaluation_Methodology.pdf
        # VoLTE uses the VoIP model
        # embb_general uses the video streaming model
        # urllc uses the FTP2 model
        if self.sys_clock == 0:
            for ser_name in self.ser_cat:
                ue_index = np.where(self.UE_cat == ser_name)
                ue_index_Size = ue_index[0].size
                if ser_name == 'volte':
                    self.UE_readtime[ue_index] = np.random.uniform(0,160 * 10 ** (-3),[1,ue_index_Size]) # the silence lasts 160 ms in maximum
                elif ser_name == 'embb_general':
                    tmp_readtime = np.random.pareto(1.2,[1,ue_index_Size]) * 6 * 10 ** -3
                    tmp_readtime[tmp_readtime > 12.5 * 10 ** -3] = 12.5 * 10 ** -3
                    self.UE_readtime[ue_index]  = tmp_readtime
                elif ser_name == 'urllc':
                    self.UE_readtime[ue_index]  = np.random.exponential(180* 10 ** -3,[1,ue_index_Size]) # read time is determines much smaller; the spec shows the average time is 180s, but here it is defined as 180 ms

        for ue_id in range(self.UE_max_no):
            if self.UE_readtime[ue_id] <= 0:
                if self.UE_buffer[:,ue_id].size - np.count_nonzero(self.UE_buffer[:,ue_id]) != 0: # The buffer is not full
                    buf_ind = np.where(self.UE_buffer[:,ue_id] == 0)[0][0]
                    if self.UE_cat[ue_id] == 'volte':
                        self.UE_buffer[buf_ind,ue_id] = 40 * 8
                        self.UE_readtime[ue_id] = np.random.uniform(0,160 * 10 ** (-3),1)
                    elif self.UE_cat[ue_id] == 'embb_general':
                        tmp_buffer_size = np.random.pareto(1.2,1) * 800 
                        if tmp_buffer_size > 2000:
                            tmp_buffer_size = 2000
                        # tmp_buffer_size = np.random.choice([1*8*10**6, 2*8*10**6, 3*8*10**6, 4*8*10**6, 5*8*10**6])
                        self.UE_buffer[buf_ind,ue_id] = tmp_buffer_size
                        self.UE_readtime[ue_id] = np.random.pareto(1.2,[1,1]) * 6 * 10 ** -3
                        if self.UE_readtime[ue_id] > 12.5 * 10 ** -3:
                            self.UE_readtime[ue_id] = 12.5 * 10 ** -3 
                    elif self.UE_cat[ue_id] == 'urllc':
                        #tmp_buffer_size = np.random.lognormal(14.45,0.35,[1,1])
                        # if tmp_buffer_size > 5 * 10 **6:
                        #      tmp_buffer_size > 5 * 10 **6
                        # tmp_buffer_size = np.random.choice([6.4*8*10**3, 12.8*8*10**3, 19.2*8*10**3, 25.6*8*10**3, 32*8*10**3])
                        tmp_buffer_size = np.random.choice([0.3*8*10**6, 0.4*8*10**6, 0.5*8*10**6, 0.6*8*10**6, 0.7*8*10**6])
                        self.UE_buffer[buf_ind,ue_id] = tmp_buffer_size
                        self.UE_readtime[ue_id]  = np.random.exponential(180* 10 ** -3,[1,1]) # read time is determines much smaller; the spec shows the average time is 180s, but here it is defined as 180 ms
                    self.tx_pkt_no[self.ser_cat.index(self.UE_cat[ue_id])] += 1
                    self.UE_buffer_backup[buf_ind,ue_id] = self.UE_buffer[buf_ind,ue_id]
                    
            else:
                self.UE_readtime[ue_id] -= self.time_subframe
                
        self.sys_clock += self.time_subframe
        self.sys_clock = round(self.sys_clock,4)
       
    def get_state(self):
        state = self.tx_pkt_no
        return state
        
    def store_reward(self,rate):
        # print("rete :" ,rate)
        se = np.zeros(len(self.ser_cat))
        ee = np.zeros(len(self.ser_cat))
        sys_rate_frame = 0
        for ser_name in self.ser_cat:
            ser_index = self.ser_cat.index(ser_name)
            ue_index_ = np.where(self.UE_cat == ser_name)
            allo_band = np.sum(self.UE_band[ue_index_])
            sum_rate = np.sum(rate[ue_index_])
            if allo_band != 0:
                sys_rate_frame += sum_rate
                se[ser_index] = sum_rate/allo_band
                ee[ser_index] = se[ser_index]/10**(self.BS_tx_power/10)
        
        # Calculating the system SE and EE
        self.sys_se_per_frame += sys_rate_frame/self.band_whole

        handling_latency = 2 * 10 ** (-3)
        handling_latency = 0
        for ue_id in range(self.UE_max_no): 
            for i in range(self.UE_latency[:,ue_id].size):
                if (self.UE_buffer[i,ue_id] == 0) & (self.UE_latency[i,ue_id] != 0): 
                    if self.UE_cat[ue_id] == 'volte': 
                        cat_index = self.ser_cat.index('volte')   
                        if (self.UE_latency[i,ue_id] == self.time_subframe):
                            if (rate[ue_id] >= 51 * 10 ** 3) & (self.UE_latency[i,ue_id] < 10 * 10 **(-3) - handling_latency): 
                                # print("in")
                                self.succ_tx_pkt_no[cat_index] += 1
                                # print("in")
                        else:
                            if (self.UE_buffer_backup[i,ue_id]/self.UE_latency[i,ue_id] >= 51 * 10 ** 3) & (self.UE_latency[i,ue_id] < 10 * 10 **(-3) - handling_latency):
                                self.succ_tx_pkt_no[cat_index] += 1
                                # print("in")
                    elif self.UE_cat[ue_id] == 'embb_general':
                        cat_index = self.ser_cat.index('embb_general')    
                        if (self.UE_latency[i,ue_id] == self.time_subframe):
                            #if (rate[ue_id] >= 5 * 10 ** 6) & (self.UE_latency[i,ue_id] < 10 * 10 **(-3) - handling_latency):
                            if (rate[ue_id] >= 100 * 10 ** 6) & (self.UE_latency[i,ue_id] < 10 * 10 **(-3) - handling_latency):
                                self.succ_tx_pkt_no[cat_index] += 1
                                # print("in")
                        else:
                            if (self.UE_buffer_backup[i,ue_id]/self.UE_latency[i,ue_id] >= 100 * 10 ** 6) & (self.UE_latency[i,ue_id] < 10 * 10 **(-3) - handling_latency):
                                self.succ_tx_pkt_no[cat_index] += 1
                                # print("in")
                    elif self.UE_cat[ue_id] == 'urllc': 
                        cat_index = self.ser_cat.index('urllc')   
                        if (self.UE_latency[i,ue_id] == self.time_subframe):
                            if (rate[ue_id] >= 10 * 10 ** 6) & (self.UE_latency[i,ue_id] < 3 * 10 **(-3) - handling_latency):
                                self.succ_tx_pkt_no[cat_index] += 1
                                # print("in")
                        else:
                            if (self.UE_buffer_backup[i,ue_id]/self.UE_latency[i,ue_id] >= 10 * 10 ** 6) & (self.UE_latency[i,ue_id] < 3 * 10 **(-3) - handling_latency):
                                self.succ_tx_pkt_no[cat_index] += 1
                                # print("in")

    def get_reward(self):
        se_total = self.sys_se_per_frame/(self.learning_windows / self.time_subframe)
        # ee_total = se_total/10**(self.BS_tx_power/10)   

        reward = self.succ_tx_pkt_no/self.tx_pkt_no 
        print("se_total :" ,se_total)
        print("self.sys_se_per_frame" ,self.sys_se_per_frame)
        print("self.learning_windows :" ,self.learning_windows)
        print("self.time_subframe :" ,self.time_subframe)

        print("in succ_tx_pkt_no: " , self.succ_tx_pkt_no)
        print("in tx_pkt_no: " , self.tx_pkt_no)
        return reward, se_total

    def bufferClear(self):    
        latency = np.sum(self.UE_latency,axis=0)
        UE_index = np.where(latency != 0) 
        bufSize = self.UE_latency[:,0].size
        for ue_id in UE_index[0]: 
            
            buffer_ = self.UE_buffer[:,ue_id].copy()
            buffer_bk = self.UE_buffer_backup[:,ue_id].copy()
            latency_ = self.UE_latency[:,ue_id].copy()
            ind_1 = np.where(np.logical_and(buffer_ ==0 , latency_ !=0 ) )
            indSize_1 = ind_1[0].size
            if indSize_1 != 0:
                self.UE_latency[ind_1,ue_id] = np.zeros(indSize_1)
                self.UE_buffer_backup[ind_1,ue_id] = np.zeros(indSize_1)

  
            ind = np.where(np.logical_and(buffer_ !=0 , latency_ !=0 ) )
            ind = ind[0]
            indSize = ind.size 
            if indSize != 0:
                self.UE_buffer[:,ue_id] = np.zeros(bufSize)
                self.UE_latency[:,ue_id] = np.zeros(bufSize)
                self.UE_buffer_backup[:,ue_id] = np.zeros(bufSize)
                self.UE_buffer[:indSize,ue_id] = buffer_[ind]
                self.UE_latency[:indSize,ue_id] = latency_[ind]
                self.UE_buffer_backup[:indSize,ue_id] = buffer_bk[ind]
            
    def countReset(self):
        self.tx_pkt_no = np.zeros(len(self.ser_cat))
        self.succ_tx_pkt_no = np.zeros(len(self.ser_cat))
        self.sys_se_per_frame = np.zeros(1)  
        self.UE_buffer = np.zeros(self.UE_buffer.shape)
        self.UE_buffer_backup = np.zeros(self.UE_buffer.shape)
        self.UE_latency = np.zeros(self.UE_buffer.shape)
    
        
def bufferUpdate(buffer,rate,time_subframe):    
    bSize = buffer.size
    for i in range(bSize):
        if buffer[i] >= rate * time_subframe:
            buffer[i] -= rate * time_subframe
            rate = 0
            break
        else:
            rate_ = buffer[i]
            buffer[i] = 0
            rate -= rate_
    return buffer

def latencyUpdate(latency,buffer,time_subframe):
    lSize = latency.size
    for i in range(lSize):
        if buffer[i]!=0:
            latency[i] += time_subframe
    return latency    



def state_update(state, ser_cat):
    discrete_state = np.zeros(state.shape)
    if state.all() == 0:
        return discrete_state
    for ser_name in ser_cat:
        ser_index = ser_cat.index(ser_name)
        discrete_state[ser_index] = state[ser_index]
    discrete_state = (discrete_state-discrete_state.mean())/discrete_state.std()
    return discrete_state
def calc_reward(qoe, se, threshold):
    utility = np.matmul(qoe_weight, qoe.reshape((3, 1))) + se_weight * se
    print("utility : ", utility)
    print("qoe: ", qoe)
    print("se :", se)
    # if threshold > 5.5:
    #     threshold = 5.5
    threshold=12
    if utility < threshold:
        reward = 0
    else:
        reward = 1
    return utility, reward

def get_act(action):
    
    if action==0:
        return [1, 1, 8]
    elif action==1:
        return [1, 2, 7]
    elif action==2:
        return [1, 3, 6]
    elif action==3:
        return [1, 4, 5]
    elif action==4:
        return [1, 5, 4]
    elif action==5:
        return [1, 6, 3]
    elif action==6: 
        return [1 ,7 ,2]
    elif action==7: 
        return [1 ,8, 1]
    elif action==8: 
        return [2 ,1, 7]
    elif action==9: 
        return [2 ,2, 6]
    elif action==10: 
        return [2 ,3, 5]
    elif action==11: 
        return [2 ,4, 4]
    elif action==12: 
        return [2 ,5, 3]
    elif action==13: 
        return [2 ,6, 2]
    elif action==14: 
        return [2 ,7, 1]
    elif action==15: 
        return [3 ,1, 6]
    elif action==16: 
        return [3 ,2, 5]
    elif action==17: 
        return [3 ,3, 4]
    elif action==18: 
        return [3 ,4, 3]
    elif action==19: 
        return [3 ,5, 2]
    elif action==20: 
        return [3 ,6, 1]
    elif action==21: 
        return [4 ,1, 5]
    elif action==22: 
        return [4 ,2, 4]
    elif action==23: 
        return [4 ,3, 3]
    elif action==24: 
        return [4 ,4, 2]
    elif action==25: 
        return [4 ,5, 1]
    elif action==26: 
        return [5 ,1, 4]
    elif action==27: 
        return [5 ,2, 3]
    elif action==28: 
        return [5 ,3, 2]
    elif action==29: 
        return [5 ,4, 1]
    elif action==30: 
        return [6 ,1, 3]
    elif action==31: 
        return [6 ,2, 2]
    elif action==32: 
        return [6 ,3, 1]
    elif action==33: 
        return [7 ,1, 2]
    elif action==34: 
        return [7 ,2, 1]
    

    

