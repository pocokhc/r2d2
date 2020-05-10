import gym
from keras.optimizers import Adam

import traceback

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.rainbow import Rainbow
from src.processor import PendulumProcessorForDQN
from src.image_model import DQNImageModel
from src.memory import *
from src.policy import *
from src.callbacks import *
from src.common import InputType, LstmType, DuelingNetwork, seed_everything, LoggerType


seed_everything(42)


def main(mode):
    
    ENV_NAME = "MountainCar-v0"

    env = gym.make(ENV_NAME)
    weight_file = "tmp/mountaincar_weight.h5"
    os.makedirs(os.path.dirname(weight_file), exist_ok=True)

    # ゲーム情報
    print("action_space      : " + str(env.action_space))
    print("observation_space : " + str(env.observation_space))
    print("reward_range      : " + str(env.reward_range))

    processor = None
    input_shape = env.observation_space.shape
    input_type = InputType.VALUES
    image_model = None
    enable_rescaling = True
    warmup = 50_000

    kwargs={
        "input_shape": input_shape, 
        "input_type": input_type,
        "nb_actions": env.action_space.n, 
        "optimizer": Adam(lr=0.0001),
        "metrics": [],

        "image_model": image_model,
        "input_sequence": 4,         # 入力フレーム数
        "dense_units_num": 64,       # dense層のユニット数
        "enable_dueling_network": True,
        "dueling_network_type": DuelingNetwork.AVERAGE,  # dueling networkで使うアルゴリズム
        "lstm_type": LstmType.STATELESS,           # 使用するLSTMアルゴリズム
        "lstm_units_num": 64,             # LSTMのユニット数
        "lstm_ful_input_length": 2,       # ステートフルLSTMの入力数

        # train/action関係
        "memory_warmup_size": warmup,    # 初期のメモリー確保用step数(学習しない)
        "target_model_update": 1000,  # target networkのupdate間隔
        "action_interval": 1,       # アクションを実行する間隔
        "train_interval": 1,        # 学習する間隔
        "batch_size": 32,     # batch_size
        "gamma": 0.99,        # Q学習の割引率
        "enable_double_dqn": True,
        "enable_rescaling": enable_rescaling,   # rescalingを有効にするか
        "rescaling_epsilon": 0.001,  # rescalingの定数
        "priority_exponent": 0.9,   # priority優先度
        "burnin_length": 1,        # burn-in期間
        "reward_multisteps": 3,    # multistep reward

        # その他
        "processor": processor,
        "action_policy": AnnealingEpsilonGreedy(
            initial_epsilon=1.0,      # 初期ε
            final_epsilon=0.01,        # 最終状態でのε
            exploration_steps=warmup+5_000  # 初期→最終状態になるまでのステップ数
        ),
        "memory": PERRankBaseMemory(
            capacity= 100_000,
            alpha=0.9,           # PERの確率反映率
            beta_initial=0.0,    # IS反映率の初期値
            beta_steps=warmup+5_000,  # IS反映率の上昇step数
            enable_is=True,     # ISを有効にするかどうか
        )
    }

    agent = Rainbow(**kwargs)
    print(agent.model.summary())

    test_agent = Rainbow(**kwargs)
    test_env = gym.make(ENV_NAME)
    log = Logger2Stage(LoggerType.STEP,
        warmup=kwargs["memory_warmup_size"],
        interval1=1000,
        interval2=5000,
        change_count=20,
        savefile="tmp/log.json",
        test_agent=test_agent,
        test_env=test_env)

    if mode == "train":
        print("--- start ---")
        print("'Ctrl + C' is stop.")
        try:
            #agent.load_weights(weight_file)
            
            mc = rl.callbacks.ModelIntervalCheckpoint(
                filepath = weight_file + '_{step:02d}.h5',
                interval=10_000
            )

            agent.fit(env, nb_steps=kwargs["memory_warmup_size"] + 100_000, visualize=False, verbose=0, callbacks=[mc, log])
            test_env.close()

        except Exception:
            print(traceback.print_exc())

        # save
        print("weight save: " + weight_file)
        agent.save_weights(weight_file, overwrite=True)
        
    # plt
    log.drawGraph()

    #--- test
    print("weight load: " + weight_file)
    agent.load_weights(weight_file)

    # 目視用
    agent.test(env, nb_episodes=5, visualize=True)

    # 動画保存用
    movie = MovieLogger()
    agent.test(env, nb_episodes=1, visualize=False, callbacks=[movie])
    movie.save(gifname="tmp/mountaincar.gif", fps=30)
    
    env.close()


if __name__ == '__main__':
    main(mode="train")
    #main(mode="test")




