import gym
from keras.optimizers import Adam

import traceback

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.rainbow import Rainbow
from src.processor import AtariProcessor
from src.image_model import DQNImageModel
from src.memory import *
from src.policy import *
from src.callbacks import *
from src.common import InputType, LstmType, DuelingNetwork, seed_everything


seed_everything(43)


def main(mode):
    
    ENV_NAME = "BreakoutDeterministic-v4"

    env = gym.make(ENV_NAME)
    processor = AtariProcessor(is_clip=False, max_steps=1000)
    enable_rescaling = True

    weight_file = "tmp/breakout_weight.h5"
    os.makedirs(os.path.dirname(weight_file), exist_ok=True)

    kwargs={
        "input_shape": processor.image_shape, 
        "input_type": InputType.GRAY_2ch,
        "nb_actions": env.action_space.n, 
        "optimizer": Adam(lr=0.0001),
        "metrics": [],

        "image_model": DQNImageModel(),
        "input_sequence": 4,         # 入力フレーム数
        "dense_units_num": 256,       # dense層のユニット数
        "enable_dueling_network": True,
        "dueling_network_type": DuelingNetwork.AVERAGE,  # dueling networkで使うアルゴリズム
        "lstm_type": LstmType.STATELESS,           # 使用するLSTMアルゴリズム
        "lstm_units_num": 128,             # LSTMのユニット数
        "lstm_ful_input_length": 1,       # ステートフルLSTMの入力数

        # train/action関係
        "memory_warmup_size": 50_000,    # 初期のメモリー確保用step数(学習しない)
        "target_model_update": 10_000,  # target networkのupdate間隔
        "action_interval": 1,       # アクションを実行する間隔
        "train_interval": 1,        # 学習する間隔
        "batch_size": 16,     # batch_size
        "gamma": 0.99,        # Q学習の割引率
        "enable_double_dqn": True,
        "enable_rescaling": enable_rescaling,   # rescalingを有効にするか
        "rescaling_epsilon": 0.001,  # rescalingの定数
        "priority_exponent": 0.9,   # priority優先度
        "burnin_length": 2,        # burn-in期間
        "reward_multisteps": 3,    # multistep reward

        # その他
        "processor": processor,
        "action_policy": AnnealingEpsilonGreedy(
            initial_epsilon=1.0,      # 初期ε
            final_epsilon=0.01,        # 最終状態でのε
            exploration_steps=1_000_000  # 初期→最終状態になるまでのステップ数
        ),
        "memory": PERRankBaseMemory(
            capacity= 50_000,
            alpha=0.9,           # PERの確率反映率
            beta_initial=0.0,    # IS反映率の初期値
            beta_steps=1_000_000,  # IS反映率の上昇step数
            enable_is=True,     # ISを有効にするかどうか
        )
    }

    agent = Rainbow(**kwargs)
    #print(agent.model.summary())

    test_agent = Rainbow(**kwargs)
    test_env = gym.make(ENV_NAME)
    log = Logger2Stage(LoggerType.STEP,
        warmup=kwargs["memory_warmup_size"],
        interval1=200,
        interval2=10_000,
        change_count=5,
        savefile="tmp/log.json",
        test_agent=test_agent,
        test_env=test_env,
        test_episodes=1  # 乱数なし
    )

    if mode == "train":
        print("--- start ---")
        print("'Ctrl + C' is stop.")
        try:
            #agent.load_weights(weight_file, load_memory=True)

            mc = ModelIntervalCheckpoint(
                filepath = weight_file + '_{step:02d}.h5',
                interval=100_000,
                save_memory=True,
            )

            #agent.fit(env, nb_steps=100_000, visualize=False, verbose=0, callbacks=[mc, log])
            agent.fit(env, nb_steps=1_750_000, visualize=False, verbose=0, callbacks=[mc, log])
            test_env.close()

        except Exception:
            print(traceback.print_exc())
        
        # save
        print("weight save: " + weight_file)
        agent.save_weights(weight_file, overwrite=True, save_memory=True)
    
    # plt
    log.drawGraph()
    
    #--- test
    print("weight load: " + weight_file)
    agent.load_weights(weight_file)
    
    agent.test(env, nb_episodes=1, visualize=True, verbose=1)

    # 動画保存用
    movie = MovieLogger()
    conv = ConvLayerView(agent)
    agent.test(env, nb_episodes=1, visualize=False, callbacks=[movie, conv])
    movie.save(gifname="breakout1.gif", fps=30)
    conv.save(grad_cam_layers=["conv_1", "conv_2", "conv_3"], add_adv_layer=True, add_val_layer=True, end_frame=200, gifname="breakout2.gif", fps=10)

    env.close()


if __name__ == '__main__':
    main(mode="train")
    #main(mode="test")




