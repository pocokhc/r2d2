import gym
from keras.optimizers import Adam

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.r2d2 import R2D2, Actor
from src.r2d2_callbacks import *
from src.processor import PendulumProcessorForDQN
from src.image_model import DQNImageModel
from src.memory import *
from src.policy import *
from src.common import InputType, LstmType, DuelingNetwork, seed_everything, LoggerType
from src.callbacks import ConvLayerView, MovieLogger


seed_everything(42)
ENV_NAME = "Pendulum-v0"


class MyActor(Actor):
    def getPolicy(self, actor_index, actor_num):
        return EpsilonGreedy(0.1)

    def fit(self, index, agent):
        env = gym.make(ENV_NAME)
        agent.fit(env, visualize=False, verbose=0)
        env.close()

class MyActor1(MyActor):
    def getPolicy(self, actor_index, actor_num):
        return EpsilonGreedy(0.01)

class MyActor2(MyActor):
    def getPolicy(self, actor_index, actor_num):
        return EpsilonGreedy(0.1)


def main(mode):

    env = gym.make(ENV_NAME)

    image = False
    if image:
        processor = PendulumProcessorForDQN(enable_image=True)
        input_shape = processor.image_shape
        input_type = InputType.GRAY_2ch
        image_model = DQNImageModel()
        enable_rescaling = False
    else:
        processor = PendulumProcessorForDQN(enable_image=False)
        input_shape = env.observation_space.shape
        input_type = InputType.VALUES
        image_model = None
        enable_rescaling = False
    
    kwargs = {
        "input_shape": input_shape,
        "input_type": input_type,
        "nb_actions": processor.nb_actions, 
        "optimizer": Adam(lr=0.0001),
        "metrics": [],

        "image_model": image_model,
        "input_sequence": 4,             # 入力フレーム数
        "dense_units_num": 64,           # Dense層のユニット数
        "enable_dueling_network": True,  # dueling_network有効フラグ
        "dueling_network_type": DuelingNetwork.AVERAGE,   # dueling_networkのアルゴリズム
        "lstm_type": LstmType.STATELESS,  # LSTMのアルゴリズム
        "lstm_units_num": 64,            # LSTM層のユニット数
        "lstm_ful_input_length": 2,      # ステートフルLSTMの入力数

        # train/action関係
        "remote_memory_warmup_size": 200,  # 初期のメモリー確保用step数(学習しない)
        "target_model_update": 1000,  #  target networkのupdate間隔
        "action_interval": 1,         # アクションを実行する間隔
        "batch_size": 16,
        "gamma": 0.997,             # Q学習の割引率
        "enable_double_dqn": True,  # DDQN有効フラグ
        "enable_rescaling": enable_rescaling,    # rescalingを有効にするか(priotrity)
        "rescaling_epsilon": 0.001,  # rescalingの定数
        "priority_exponent": 0.9,   # priority優先度
        "burnin_length": 1,          # burn-in期間
        "reward_multisteps": 3,  # multistep reward

        # その他
        "processor": processor,
        "actors": [MyActor1, MyActor2],
        "remote_memory": PERRankBaseMemory(
            capacity= 50_000,
            alpha=0.9,           # PERの確率反映率
            beta_initial=0.0,    # IS反映率の初期値
            beta_steps=10_000,  # IS反映率の上昇step数
            enable_is=True,     # ISを有効にするかどうか
        ),

        # actor 関係
        "actor_model_sync_interval": 200,  # learner から model を同期する間隔
    }
    
    #--- R2D2
    manager = R2D2(**kwargs)

    test_env = gym.make(ENV_NAME)
    log = Logger2Stage(
        interval1=10,
        interval2=60,
        change_count=20,
        savedir="tmp",
        test_actor=MyActor,
        test_env=test_env)
    
    if mode == "train":
        print("--- start ---")
        print("'Ctrl + C' is stop.")

        save_manager = SaveManager(
            save_dirpath="tmp",
            is_load=False,
            save_memory=True,
            checkpoint=True,
            checkpoint_interval=2000,
            verbose=0
        )

        manager.train(
            nb_trains=15_000,
            callbacks=[save_manager, log],
        )

    log.drawGraph()

    # 訓練結果を見る
    agent = manager.createTestAgent(MyActor, "tmp/last/learner.dat")
    agent.test(env, nb_episodes=5, visualize=True)

    # 動画保存用
    if image:
        conv = ConvLayerView(agent)
        agent.test(env, nb_episodes=1, visualize=False, callbacks=[conv])
        conv.save(grad_cam_layers=["conv_1", "conv_2", "conv_3"], add_adv_layer=True, add_val_layer=True, end_frame=200, gifname="tmp/pendulum.gif", fps=10)
    else:
        movie = MovieLogger()
        agent.test(env, nb_episodes=1, visualize=False, callbacks=[movie])
        movie.save(gifname="tmp/pendulum.gif", fps=30)
    
    env.close()


if __name__ == '__main__':
    main(mode="train")
    #main(mode="test")

