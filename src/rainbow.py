
import rl
import rl.core
import keras
from keras.layers import *
from keras.models import Model
from keras.models import model_from_json
from keras.utils import CustomObjectScope

import os
import pickle

from .common import *


#---------------------------------------------------
# Rainbow
#---------------------------------------------------
class Rainbow(rl.core.Agent):
    def __init__(self,
            input_shape,
            input_type,
            nb_actions,
            memory,
            action_policy,
            optimizer,

            metrics=[],
            image_model=None,    # imegeモデルを指定
            input_sequence=4,    # 入力フレーム数
            dense_units_num=512, # Dense層のユニット数
            enable_dueling_network=True,                  # dueling network有効フラグ
            dueling_network_type=DuelingNetwork.AVERAGE,  # dueling networkで使うアルゴリズム
            lstm_type=LstmType.NONE,  # LSTM有効フラグ
            lstm_units_num=512,       # LSTMのユニット数
            lstm_ful_input_length=1,  # ステートフルLSTMの入力数

            # burn-in有効時の設定
            burnin_length=4,          # burn-in期間
            enable_rescaling=False,   # rescalingを有効にするか
            rescaling_epsilon=0.001,  # rescalingの定数
            priority_exponent=0.9,    # シーケンス長priorityを計算する際のη
            
            # train関係
            batch_size=32,            # batch_size
            memory_warmup_size=50000, # 初期メモリー確保用step数(学習しない)
            target_model_update=500,  # target networkのupdate間隔
            enable_double_dqn=True,   # DDQN有効フラグ
            action_interval=4,    # アクションを実行する間隔
            train_interval=4,     # 学習間隔
            gamma=0.99,           # Q学習の割引率
            reward_multisteps=3,  # multistep reward

            processor=None
        ):
        super(Rainbow, self).__init__(processor)
        self.compiled = False  # super()

        #--- check
        if lstm_type == LstmType.STATEFUL:
            self.burnin_length = burnin_length
        else:
            self.burnin_length = 0

        assert memory.capacity > batch_size, "Memory capacity is small.(Larger than batch size)"
        assert memory_warmup_size > batch_size, "Warmup steps is few.(Larger than batch size)"

        if image_model is None:
            assert input_type == InputType.VALUES
        else:
            assert input_type == InputType.GRAY_2ch or input_type == InputType.GRAY_3ch or input_type == InputType.COLOR

            # 画像入力の制約
            # LSTMを使う場合: 画像は(w,h,ch)で入力できます。
            # LSTMを使わない場合：
            #   input_sequenceが1：全て使えます。
            #   input_sequenceが1以外：GRAY_2ch のみ使えます。
            if lstm_type == LstmType.NONE and input_sequence != 1:
                assert (input_type == InputType.GRAY_2ch), "input_iimage can use GRAY_2ch."

        #---
        self.input_shape = input_shape
        self.image_model = image_model
        self.input_type = input_type

        self.nb_actions = nb_actions
        self.input_sequence = input_sequence
        self.memory_warmup_size = memory_warmup_size
        self.target_model_update = target_model_update
        self.action_interval = action_interval
        self.train_interval = train_interval
        self.gamma = gamma
        self.batch_size = batch_size
        assert reward_multisteps > 0, "'reward_multisteps' is 1 or more."
        self.reward_multisteps = reward_multisteps
        self.dense_units_num = dense_units_num

        self.lstm_units_num = lstm_units_num
        self.enable_rescaling = enable_rescaling
        self.rescaling_epsilon = rescaling_epsilon
        self.priority_exponent = priority_exponent
        self.lstm_type = lstm_type

        self.optimizer = optimizer
        self.metrics = metrics

        self.memory = memory
        self.action_policy = action_policy
        
        self.lstm_ful_input_length = lstm_ful_input_length
        
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_network_type = dueling_network_type
        
        self.model = self.build_compile_model()  # Q network
        model_json = self.model.to_json()
        self.target_model = model_from_json(model_json)
        self.action_policy.compile(model_json)

        if self.lstm_type == LstmType.STATEFUL:
            self.lstm = self.model.get_layer("lstm")
            self.target_lstm = self.target_model.get_layer("lstm")

        self.compiled = True  # super

        self.local_step = 0


    def reset_states(self):  # override
        self.repeated_action = 0
        
        if self.lstm_type == LstmType.STATEFUL:
            multi_len = self.reward_multisteps + self.lstm_ful_input_length - 1
            self.recent_actions = [ 0 for _ in range(multi_len + 1)]
            self.recent_rewards = [ 0 for _ in range(multi_len)]
            self.recent_rewards_multistep = [ 0 for _ in range(self.lstm_ful_input_length)]
            tmp = self.burnin_length + self.input_sequence + multi_len
            self.recent_observations = [
                np.zeros(self.input_shape) for _ in range(tmp)
            ]
            tmp = self.burnin_length + multi_len + 1
            self.recent_observations_wrap = [
                [np.zeros(self.input_shape) for _ in range(self.input_sequence)] for _ in range(tmp)
            ]

            # hidden_state: [(batch_size, lstm_units_num), (batch_size, lstm_units_num)]
            tmp = self.burnin_length + multi_len + 1+1
            self.model.reset_states()
            self.recent_hidden_states = [
                [K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])] for _ in range(tmp)
            ]
            
        else:
            self.recent_actions = [ 0 for _ in range(self.reward_multisteps+1)]
            self.recent_rewards = [ 0 for _ in range(self.reward_multisteps)]
            self.recent_rewards_multistep = 0
            self.recent_observations = [
                np.zeros(self.input_shape) for _ in range(self.input_sequence + self.reward_multisteps)
            ]

    def build_compile_model(self):

        if self.lstm_type == LstmType.STATEFUL:
            # input(batch_size, timesteps, shape)
            c = input_ = Input(batch_shape=(self.batch_size, self.input_sequence) + self.input_shape)
        else:
            # input(input_sequence, shape)
            c = input_ = Input(shape=(self.input_sequence,) + self.input_shape)
        
        
        if self.image_model is None:
            # input not image
            if self.lstm_type == LstmType.NONE:
                c = Flatten()(c)
            else:
                c = TimeDistributed(Flatten())(c)
        else:
            # input image
            if self.lstm_type == LstmType.NONE:
                enable_lstm = False
                if self.input_type == InputType.GRAY_2ch:
                    # (input_seq, w, h) ->(w, h, input_seq)
                    c = Permute((2, 3, 1))(c)

            elif self.lstm_type == LstmType.STATELESS or self.lstm_type == LstmType.STATEFUL:
                enable_lstm = True
                if self.input_type == InputType.GRAY_2ch:
                    # (time steps, w, h) -> (time steps, w, h, ch)
                   c = Reshape((self.input_sequence, ) + self.input_shape + (1,) )(c)
                
            else:
                raise ValueError('lstm_type is not undefined')
            c = self.image_model.create_image_model(c, enable_lstm)

        # lstm layer
        if self.lstm_type == LstmType.STATELESS:
            c = LSTM(self.lstm_units_num, name="lstm")(c)
        elif self.lstm_type == LstmType.STATEFUL:
            c = LSTM(self.lstm_units_num, stateful=True, name="lstm")(c)

        # dueling network
        if self.enable_dueling_network:
            # value
            v = Dense(self.dense_units_num, activation="relu")(c)
            v = Dense(1, name="v")(v)

            # advance
            adv = Dense(self.dense_units_num, activation='relu')(c)
            adv = Dense(self.nb_actions, name="adv")(adv)

            # 連結で結合
            c = Concatenate()([v,adv])
            if self.dueling_network_type == DuelingNetwork.AVERAGE:
                c = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(self.nb_actions,))(c)
            elif self.dueling_network_type == DuelingNetwork.MAX:
                c = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], axis=1, keepdims=True), output_shape=(self.nb_actions,))(c)
            elif self.dueling_network_type == DuelingNetwork.NAIVE:
                c = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(self.nb_actions,))(c)
            else:
                raise ValueError('dueling_network_type is not undefined')
        else:
            c = Dense(self.dense_units_num, activation="relu")(c)
            c = Dense(self.nb_actions, activation="linear", name="adv")(c)
        
        model = Model(input_, c)
        model.compile(loss=clipped_error_loss, optimizer=self.optimizer, metrics=self.metrics)
        self.compiled = True  # super

        return model

    def compile(self, optimizer, metrics=[]):  # override
        self.compiled = True  # super

    def save_weights(self, filepath, overwrite=False, save_memory=False):  # override
        if overwrite or not os.path.isfile(filepath):
            d = {
                "weights": self.model.get_weights(),
                "policy": self.action_policy.get_weights(),
                "step": self.local_step,
            }
            with open(filepath, 'wb') as f:
                pickle.dump(d, f)
            
            # memory
            if save_memory:
                d = self.memory.get_memorys()
                with open(filepath + ".mem", 'wb') as f:
                    pickle.dump(d, f)


    def load_weights(self, filepath, load_memory=False):  # override
        if not os.path.isfile(filepath):
            return
        with open(filepath, 'rb') as f:
            d = pickle.load(f)
        self.model.set_weights(d["weights"])
        self.target_model.set_weights(d["weights"])
        self.action_policy.set_weights(d["policy"])
        self.local_step = d["step"]

        # memory
        if load_memory:
            filepath = filepath + ".mem"
            if os.path.isfile(filepath):
                with open(filepath, 'rb') as f:
                    d = pickle.load(f)
                self.memory.set_memorys(d)

    def forward(self, observation):  # override
        # observation
        self.recent_observations.pop(0)
        self.recent_observations.append(observation)

        if self.lstm_type == LstmType.STATEFUL:
            self.recent_observations_wrap.pop(0)
            self.recent_observations_wrap.append(self.recent_observations[-self.input_sequence:])

        # tmp
        self._qvals = None
        self._state1 = self.recent_observations[-self.input_sequence:]
        self._state1_np = np.asarray(self._state1)

        # 学習(次の状態が欲しいのでforwardで学習)
        if self.training:
            self.forward_train()

        # 状態の更新
        if self.lstm_type == LstmType.STATEFUL:
            self.lstm.reset_states(self.recent_hidden_states[-1])

            # hidden_state を更新しつつQ値も取得
            state = self._state1_np
            pred_state = np.full((self.batch_size,)+state.shape, state)  # batchサイズ分増やす
            self._qvals = self.model.predict(pred_state, batch_size=self.batch_size)[0]
            
            hidden_state = [K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])]
            self.recent_hidden_states.pop(0)
            self.recent_hidden_states.append(hidden_state)
        
        # フレームスキップ(action_interval毎に行動を選択する)
        action = self.repeated_action
        if self.step % self.action_interval == 0:

            # 行動を決定
            if self.training:
                # training中は action policyに従う
                action = self.action_policy.select_action(self)
            else:
                # テスト中またはNoisyNet中の場合
                action = np.argmax(self.get_qvals())

            # リピート用
            self.repeated_action = action
        
        # アクション保存
        self.recent_actions.pop(0)
        self.recent_actions.append(action)
        
        return action
    

    def get_qvals(self):
        if self.lstm_type == LstmType.STATEFUL:
            return self._qvals
        else:
            if self._qvals is None:
                self._qvals = self.model.predict(
                    self._state1_np[np.newaxis,:], batch_size=1)[0]
            return self._qvals

    def get_state(self):
        return self._state1_np

    def get_prev_state(self):
        if self.lstm_type == LstmType.STATEFUL:
            observation = np.asarray(self.recent_observations_wrap[-self.reward_multisteps-1])
            action = self.recent_actions[-self.reward_multisteps-1]
            reward = self.recent_rewards_multistep[-self.reward_multisteps]
        else:
            observation = np.asarray(self.recent_observations[:self.input_sequence])
            action = self.recent_actions[0]
            reward = self.recent_rewards_multistep
        return (observation, action, reward)

    # 長いのでこちらに
    def forward_train(self):
        
        if self.lstm_type == LstmType.STATEFUL:
            self.memory.add((
                self.recent_observations_wrap[:],
                self.recent_actions[0:self.lstm_ful_input_length],
                self.recent_rewards_multistep[:],
                self.recent_hidden_states[0]))

        else:
            self.memory.add((
                self.recent_observations[:self.input_sequence],
                self.recent_actions[0],
                self.recent_rewards_multistep, 
                self._state1))

        # 初期のReplay Memoryの確保、学習しない。
        if len(self.memory) <= self.memory_warmup_size:
            return
        
        # 学習の更新間隔
        if self.step % self.train_interval != 0:
            return

        # memory から優先順位に基づき状態を取得
        (indexes, batchs, weights) = self.memory.sample(self.batch_size, self.local_step)
        
        # 学習(長いので関数化)
        if self.lstm_type == LstmType.STATEFUL:
            self.train_model_ful(indexes, batchs, weights)
        else:
            self.train_model(indexes, batchs, weights)

    # ノーマルの学習
    def train_model(self, indexes, batchs, weights):
        state0_batch = []
        action_batch = []
        reward_batch = []
        state1_batch = []
        for i, batch in enumerate(batchs):
            state0_batch.append(batch[0])
            action_batch.append(batch[1])
            reward_batch.append(batch[2])
            state1_batch.append(batch[3])
        state0_batch = np.asarray(state0_batch)
        state1_batch = np.asarray(state1_batch)
    
        # 更新用に現在のQネットワークを出力(Q network)
        state0_qvals = self.model.predict(state0_batch, self.batch_size)

        if self.enable_double_dqn:
            # TargetNetworkとQNetworkのQ値を出す
            state1_qvals_model = self.model.predict(state1_batch, self.batch_size)
            state1_qvals_target = self.target_model.predict(state1_batch, self.batch_size)
        else:
            # 次の状態のQ値を取得(target_network)
            state1_qvals_target = self.target_model.predict(state1_batch, self.batch_size)

        for i in range(self.batch_size):
            if self.enable_double_dqn:
                action = state1_qvals_model[i].argmax()  # modelからアクションを出す
                maxq = state1_qvals_target[i][action]  # Q値はtarget_modelを使って出す
            else:
                maxq = state1_qvals_target[i].max()
            
            # priority計算
            q0 = state0_qvals[i][action_batch[i]]
            td_error = reward_batch[i] + (self.gamma ** self.reward_multisteps) * maxq - q0
            priority = abs(td_error)
            
            # Q値の更新
            state0_qvals[i][action_batch[i]] += td_error * weights[i]

            # priorityを更新
            self.memory.update(indexes[i], batchs[i], priority)

        # 学習
        self.model.train_on_batch(state0_batch, state0_qvals)
    

    # ステートフルLSTMの学習
    def train_model_ful(self, indexes, batchs, weights):

        hidden_s0 = []
        hidden_s1 = []
        for batch in batchs:
            # batchサイズ分あるけどすべて同じなので0番目を取得
            hidden_s0.append(batch[3][0][0])
            hidden_s1.append(batch[3][1][0])
        hidden_states = [np.asarray(hidden_s0), np.asarray(hidden_s1)]

        # init hidden_state
        self.lstm.reset_states(hidden_states)
        self.target_lstm.reset_states(hidden_states)

        # predict
        hidden_states_arr = []
        if self.burnin_length == 0:
            hidden_states_arr.append(hidden_states)
        state_batch_arr = []
        model_qvals_arr = []
        target_qvals_arr = []
        prioritys = [ [] for _ in range(self.batch_size)]
        for seq_i in range(self.burnin_length + self.reward_multisteps + self.lstm_ful_input_length):

            # state
            state_batch = [ batch[0][seq_i] for batch in batchs ]
            state_batch = np.asarray(state_batch)
            
            # hidden_state更新およびQ値取得
            model_qvals = self.model.predict(state_batch, self.batch_size)
            target_qvals = self.target_model.predict(state_batch, self.batch_size)

            # burnin-1
            if seq_i < self.burnin_length-1:
                continue
            hidden_states_arr.append([K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])])

            # burnin
            if seq_i < self.burnin_length:
                continue

            state_batch_arr.append(state_batch)
            model_qvals_arr.append(model_qvals)
            target_qvals_arr.append(target_qvals)

        # train
        for seq_i in range(self.lstm_ful_input_length):

            # state0 の Qval (multistep前)
            state0_qvals = model_qvals_arr[seq_i]
            
            # batch
            for batch_i in range(self.batch_size):

                # maxq
                if self.enable_double_dqn:
                    action = model_qvals_arr[seq_i+self.reward_multisteps][batch_i].argmax()  # modelからアクションを出す
                    maxq = target_qvals_arr[seq_i+self.reward_multisteps][batch_i][action]  # Q値はtarget_modelを使って出す
                else:
                    maxq = target_qvals_arr[seq_i+self.reward_multisteps][batch_i].max()

                # priority
                batch_action = batchs[batch_i][1][seq_i]
                q0 = state0_qvals[batch_i][batch_action]
                reward = batchs[batch_i][2][seq_i]
                td_error = reward + (self.gamma ** self.reward_multisteps) * maxq - q0
                priority = abs(td_error)
                prioritys[batch_i].append(priority)

                # Q値の更新
                state0_qvals[batch_i][batch_action] += td_error * weights[batch_i]

            # train
            self.lstm.reset_states(hidden_states_arr[seq_i])
            self.model.train_on_batch(state_batch_arr[seq_i], state0_qvals)
            
        # priority update
        for batch_i, batch in enumerate(batchs):
            priority = self.priority_exponent * np.max(prioritys[batch_i]) + (1-self.priority_exponent) * np.average(prioritys[batch_i])
            self.memory.update(indexes[batch_i], batch, priority)
                

    def backward(self, reward, terminal):  # override
        # terminal は env が終了状態ならTrue
        self.local_step += 1
        if not self.training:
            return []
        
        # 報酬の保存
        self.recent_rewards.pop(0)
        self.recent_rewards.append(reward)

        # multi step learning の計算
        _tmp = 0
        for i in range(-self.reward_multisteps, 0):
            r = self.recent_rewards[i]
            _tmp += r * (self.gamma ** i)
        
        # rescaling
        if self.enable_rescaling:
            _tmp = rescaling(_tmp)

        if self.lstm_type == LstmType.STATEFUL:
            self.recent_rewards_multistep.pop(0)
            self.recent_rewards_multistep.append(_tmp)
        else:
            self.recent_rewards_multistep = _tmp

        # 一定間隔でtarget modelに重さをコピー
        if self.step % self.target_model_update == 0:
            self.target_model.set_weights(self.model.get_weights())

        return []
    
    @property
    def layers(self):  #override
        return self.model.layers[:]
