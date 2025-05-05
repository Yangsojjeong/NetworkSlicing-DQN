
import gymnasium
from RL_brain import DuelingDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from UREnv import MyEnv

tf.reset_default_graph()   # bastan va khali kardane hafezeye tf haye ghabli
MEMORY_SIZE = 100
ACTION_SPACE = 34   #행동공간 크기
Numb_features=3  #상태 특성 수
mode="WD"
D=10
E=10
#env=env_network(mode,D,E)
env = MyEnv(learning_windows=2000)
# env = env.unwrapped
# env.seed(1)

sess = tf.Session()  #Tensorflow 모델을 학습하려면 세션 필요
with tf.variable_scope('natural'):
    natural_DQN = DuelingDQN(
        n_actions=ACTION_SPACE, n_features=Numb_features, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, sess=sess, dueling=False)       # 일반 DQN

with tf.variable_scope('dueling'):
    dueling_DQN = DuelingDQN(
        n_actions=ACTION_SPACE, n_features=Numb_features, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, sess=sess, dueling=True, output_graph=True)    #Dueling DQN

sess.run(tf.global_variables_initializer())   #Tensorflow 변수들 초기화(안하면 오류 발생)


def train(RL):
    acc_r = [0] #누적 보상 기록용 리스트
    action_list=[] #어떤 행동을 선택했는지 저장하는 리스트
    total_steps = 0 #타임스텝 카운터
    observation = env.reset() #환경 초기화해서 첫 상태 가져오기
    #print("observation",observation)
    
    while True:
        # if total_steps-MEMORY_SIZE > 9000: env.render()

        action = RL.choose_action(observation) #현재 상태에서 행동 하나 선택
        action_list.append(action) #어떤 행동을 했는지 기록
        # print("teeeeeeest",action_list)
        # print(action_list[0])
        # print(action_list[1])
        # print(action_list[2])
        # print("action",action)
        #f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # [-2 ~ 2] float actions
        # print("**********************************")
        # print("observation",observation)
        observation_, reward, done, info = env.step(np.array([action])) #선택한 행동을 환경에 적용하고 다음 상태, 보상, 종료 여부, 추가정보 받기
        # print("observation_",observation_)
        acc_r.append(reward + acc_r[-1])  # accumulated reward
        # print("observation.shape,action,reward,observation_.shape")
        # print("#################")
        # print("observation",observation)
        # print("revar",reward)
        # print("observation_",observation_)
        RL.store_transition(observation, action, reward, observation_) #(s,a,r,s')형태로 메모리에 저장

        if total_steps > MEMORY_SIZE:
            RL.learn()  #메모리가 어느정도 쌓였으면 본격적으로 Q학습 시작

        if done:
            break

        observation = observation_ #상태 업데이트
        total_steps += 1 #카운터 증가
        print("step: ", total_steps)
    return RL.cost_his, acc_r , action_list

c_natural, r_natural , a_natural = train(natural_DQN)
print("chaaaaaangeeed")
c_dueling, r_dueling , a_dueling = train(dueling_DQN) 

plt.figure(1)
plt.plot(np.array(c_natural), c='r', label='natural')
plt.plot(np.array(c_dueling), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('cost')
plt.xlabel('training steps')
plt.grid()

plt.figure(2)
plt.plot(np.array(r_natural), c='r', label='natural')
plt.plot(np.array(r_dueling), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('accumulated reward')
plt.xlabel('training steps')
plt.grid()

plt.show()
