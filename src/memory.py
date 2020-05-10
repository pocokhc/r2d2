import numpy as np

import random
import bisect
import math

class Memory():
    """ Abstract base class for all implemented Memory. """
    def add(self, experience, priority=0):
        raise NotImplementedError()

    def update(self, idx, experience, priority):
        raise NotImplementedError()

    def sample(self, batch_size, steps):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class ReplayMemory(Memory):
    """ ReplayMemory
    https://arxiv.org/abs/1312.5602
    """

    def __init__(self, capacity):
        self.capacity= capacity
        self.index = 0
        self.buffer = []

    def add(self, experience, priority=0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = experience
        self.index = (self.index + 1) % self.capacity

    def update(self, idx, experience, priority):
        pass

    def sample(self, batch_size, steps):
        batchs = random.sample(self.buffer, batch_size)
        indexes = np.empty(batch_size)
        weights = [ 1 for _ in range(batch_size)]
        return (indexes, batchs, weights)

    def __len__(self):
        return len(self.buffer)




class _bisect_wrapper():
    def __init__(self, data):
        self.d = data
        self.priority = 0
    def __lt__(self, o):  # a<b
        return self.priority < o.priority

class PERGreedyMemory(Memory):
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.max_priority = 1

    def add(self, experience, priority=0):
        if priority == 0:
            priority = self.max_priority
        if self.capacity <= len(self.buffer):
            # 上限より多い場合は要素を削除
            self.buffer.pop(0)
        
        # priority は最初は最大を選択
        experience = _bisect_wrapper(experience)
        experience.priority = priority
        bisect.insort(self.buffer, experience)

    def update(self, idx, experience, priority):
        experience = _bisect_wrapper(experience)
        experience.priority = priority
        bisect.insort(self.buffer, experience)

        if self.max_priority < priority:
            self.max_priority = priority
    
    def sample(self, batch_size, step):
        # 取り出す(学習後に再度追加)
        batchs = [self.buffer.pop().d for _ in range(batch_size)]
        indexes = np.empty(batch_size, dtype='int')
        weights = [ 1 for _ in range(batch_size)]
        return (indexes, batchs, weights)

    def __len__(self):
        return len(self.buffer)




class SumTree():
    """
    copy from https://github.com/jaromiru/AI-blog/blob/5aa9f0b/SumTree.py
    """
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

class PERProportionalMemory(Memory):
    def __init__(self, capacity, alpha, beta_initial, beta_steps, enable_is):
        self.capacity = capacity
        self.tree = SumTree(capacity)

        self.beta_initial = beta_initial
        self.beta_steps = beta_steps
        self.enable_is = enable_is
        self.alpha = alpha

        self.size = 0
        self.max_priority = 1

    def add(self, experience, priority=0):
        if priority == 0:
            priority = self.max_priority
        priority = priority ** self.alpha
        self.tree.add(priority, experience)
        self.size += 1
        if self.size > self.capacity:
            self.size = self.capacity

    def update(self, index, experience, priority):
        priority = priority ** self.alpha
        self.tree.update(index, priority)

        if self.max_priority < priority:
            self.max_priority = priority

    def sample(self, batch_size, step):
        indexes = []
        batchs = []
        weights = np.empty(batch_size, dtype='float32')

        if self.enable_is:
            # βは最初は低く、学習終わりに1にする
            beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps
            if beta > 1:
                beta = 1
    
        total = self.tree.total()
        for i in range(batch_size):
            
            # indexesにないものを追加
            loop_over = True
            for _ in range(100):  # for safety
                r = random.random()*total
                (idx, priority, experience) = self.tree.get(r)
                if idx not in indexes:
                    loop_over = False
                    break
            #assert not loop_over

            indexes.append(idx)
            batchs.append(experience)

            if self.enable_is:
                # 重要度サンプリングを計算
                weights[i] = (self.size * priority / total) ** (-beta)
            else:
                weights[i] = 1  # 無効なら1

        if self.enable_is:
            # 安定性の理由から最大値で正規化
            weights = weights / weights.max()

        return (indexes ,batchs, weights)

    def __len__(self):
        return self.size





class _bisect_wrapper():
    def __init__(self, data):
        self.d = data
        self.priority = 0
        self.p = 0
    def __lt__(self, o):  # a<b
        return self.priority < o.priority

def rank_sum(k, a):
    return k*( 2+(k-1)*a )/2

def rank_sum_inverse(k, a):
    if a == 0:
        return k
    t = a-2 + math.sqrt((2-a)**2 + 8*a*k)
    return t/(2*a)

class PERRankBaseMemory(Memory):
    def __init__(self, capacity, alpha, beta_initial, beta_steps, enable_is):
        self.capacity = capacity
        self.buffer = []
        self.alpha = alpha
        
        self.beta_initial = beta_initial
        self.beta_steps = beta_steps
        self.enable_is = enable_is

        self.max_priority = 1

    def add(self, experience, priority=0):
        if priority == 0:
            priority = self.max_priority
        if self.capacity <= len(self.buffer):
            # 上限より多い場合は要素を削除
            self.buffer.pop(0)
        
        experience = _bisect_wrapper(experience)
        experience.priority = priority
        bisect.insort(self.buffer, experience)

    def update(self, index, experience, priority):
        experience = _bisect_wrapper(experience)
        experience.priority = priority
        bisect.insort(self.buffer, experience)

        if self.max_priority < priority:
            self.max_priority = priority


    def sample(self, batch_size, step):
        indexes = []
        batchs = []
        weights = np.empty(batch_size, dtype='float32')

        if self.enable_is:
            # βは最初は低く、学習終わりに1にする。
            beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps
            if beta > 1:
                beta = 1

        # 合計値をだす
        buffer_size = len(self.buffer)
        total = rank_sum(buffer_size, self.alpha)
        
        # index_lst
        index_lst = []
        for _ in range(batch_size):

            # index_lstにないものを追加
            for _ in range(100):  # for safety
                r = random.random()*total
                index = rank_sum_inverse(r, self.alpha)
                index = int(index)  # 整数にする(切り捨て)
                if index not in index_lst:
                    index_lst.append(index)
                    break
        #assert len(index_lst) == batch_size
        index_lst.sort()

        for i, index in enumerate(reversed(index_lst)):
            o = self.buffer.pop(index)  # 後ろから取得するのでindexに変化なし
            batchs.append(o.d)
            indexes.append(index)

            if self.enable_is:
                # 重要度サンプリングを計算
                # 確率を計算(iでの区間/total)
                r1 = rank_sum(index+1, self.alpha)
                r2 = rank_sum(index, self.alpha)
                priority = (r1-r2) / total
                w = (buffer_size * priority) ** (-beta)
                weights[i] = w
            else:
                weights[i] = 1  # 無効なら1
        
        if self.enable_is:
            # 安定性の理由から最大値で正規化
            weights = weights / weights.max()

        return (indexes, batchs, weights)

    def __len__(self):
        return len(self.buffer)

