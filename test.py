# @File  : test.py
# @Author: 沈昌力
# @Date  : 2018/5/14
# @Desc  :
from multiprocessing import Process, Queue
import os, time, random


# 写数据进程执行的代码:
def write(q):
    while True:
        for value in ['A', 'B', 'C']:
            print('Put %s to queue...queue lenght=%d' % (value, q.qsize()))
            q.put(value)

            time.sleep(random.random())


# 读数据进程执行的代码:
def read(q):
    while True:
        if not q.empty():
            value = q.get(True)
            print('Get %s from queue.' % value)
            time.sleep(random.random())
        else:
            print('Nonthing')


if __name__ == '__main__':
    # 父进程创建Queue，并传给各个子进程：
    q = Queue()
    pw = Process(target=write, args=(q,))
    pw.start()
    for i in range(6):
        pr = Process(target=read, args=(q,))
        pr.start()
        pr.join()
    # pr进程里是死循环，无法等待其结束，只能强行终止:
    print('所有数据都写入并且读完')