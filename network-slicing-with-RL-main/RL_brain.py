

import numpy as np  #numpy:수치 연산 라이브러리(np:numpy를 짧게 쓰기 위한 별칭)
import tensorflow.compat.v1 as tf    #tensorflow:구글에서 만든 머신러닝 라이브러리

np.random.seed(42)
tf.set_random_seed(42)


class DuelingDQN:
    def __init__
    (
            self,
            n_actions,  #에이전트가 선택할 수 있는 행동의 개수
            n_features,  #상태(state)의 특징 수
            learning_rate=0.001,   #학습률(얼마나 빠르게 학습할지)
            reward_decay=0.9,    #감가율(할인율)
            e_greedy=0.9,   #e-greedy 정책에서 행동 선택 시 랜덤이 아닐 확률
            replace_target_iter=200,   #몇 번 학습마다 target네트워크를 갱신할지
            memory_size=500,  #경험을 저장하는 버퍼 크기
            batch_size=32,   #학습할 때 한 번에 꺼내오는 데이터 수
            e_greedy_increment=None, #e를 조금씩 증가시킬지 여부
            output_graph=False,
            dueling=True,   #Dueling 구조를 사용할지의 여부
            sess=None,   #TensorFlow 세션
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        
        self.dueling = dueling      # decide to use dueling DQN or not
        
        self.learn_step_counter = 0    #학습이 몇 번 반복되었는지를 기록하는 변수 -> 일정 횟수마다 target네트워크를 갱신하는데 사용됨
        self.memory = np.zeros((self.memory_size, n_features*2+2))   #(s,a,r,s')형태의 경험을 저장할 배열 초기화
        self._build_net()  #신경망(정책망, 타겟망)두 개를 만듦
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)] #일정 횟수마다 두 네트워크의 파라미터를 동일하게 동기화

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []   #학습 중 발생하는 손실(loss) 저장해두는 리스트트

    def _build_net(self):         #네트워크 생성 
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):   #Q(s,a)를 계산하는 신경망망 생성
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            if self.dueling:
                # Dueling DQN  (dueling:상태의 가치V(s)와 행동별 이득A(s,a)를 분리해서 학습)
                with tf.variable_scope('Value'):
                    w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Advantage'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.A = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Q'):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
            else:   #dueling 사용안함(일반DQN)
                with tf.variable_scope('Q'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    out = tf.matmul(l1, w2) + b2

            return out

        # build evaluate_net
        tf.disable_eager_execution()  #외부 구조:네트워크와 손실함수, 학습연산 정의
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):   #평가용 네트워크 생성(eval_net)->실제 행동을 선택하고 학습되는 네트워크
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):    #손실함수 정의(q_target-q_eval 제곱의 평균)
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):   #학습연산 정의(RMSProp 옵티마이저로 손실 최소화)
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        #build target_net
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input / 다음상태(next state)입력
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer) #target_net에서 계산된 Q(s', a') -> 타겟 Q계산에 사용(학습은 안됨, 고정된 네트워크)

    def store_transition(self, s, a, r, s_):  #경험 리플레이 버퍼에 하나의 경험(s,a,r,s')을 저장
        if not hasattr(self, 'memory_counter'): # returns True if the specified object has the specified attribute, otherwise False .
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))  #하나의 경험을 만듦(s:현재상태/a:행동/r:보상/s_:다음상태)
        index = self.memory_counter % self.memory_size  #memory_size를 넘지 않도록 순환 저장 방식(ex:메모리가 500개면 501번째는 1번째를 덮어씌움)
        #print(self.memory_counter,self.memory_size,index)
        #print(self.memory.shape)
        #print(self.memory)
        
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):  #주어진 상태에서 행동 하나 선택(ε-greedy 방식)
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:  # choosing action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)  #그렇지 않으면 랜덤하게 행동 선택->탐험
        return action 

    def learn(self): #미니배치로 샘플을 뽑고 Q-Learning 학습
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')       #일정 횟수마다 target network를 eval network로 교체

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next = self.sess.run(self.q_next, feed_dict={self.s_: batch_memory[:, -self.n_features:]}) # next observation/target_net에서 다음상태(s')의 Q값 계산
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})  #eval_net에서 현재 상태(s)의 Q값 계산

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)  #벨만 방정식 기반으로 Q타겟 계산

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
