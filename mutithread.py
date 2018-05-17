# @File  : mutithread.py
# @Author: 沈昌力
# @Date  : 2018/5/15
# @Desc  :
#coding:utf-8
import queue
import threading
import time
import random
from multiprocessing.dummy import Pool as ThreadPool

q = queue.Queue(0) #当有多个线程共享一个东西的时候就可以用它了
NUM_WORKERS = 10

class MyThread(threading.Thread):

    def __init__(self,input,worktype):
       self._jobq = input
       self._work_type = worktype
       threading.Thread.__init__(self)

    def run(self):
       while True:
           if self._jobq.qsize() > 0:
               self._process_job(self._jobq.get(),self._work_type)
           else:break

    def _process_job(self, job, worktype):
       doJob(job,worktype)

def process_job(queue):
    time.sleep(random.random() * 3)
    job = queue.get()
    print("doing %d" % (job))

def doJob(job, worktype):
   time.sleep(random.random() * 3)
   print("doing %d worktype %d" %(job, worktype))

if __name__ == '__main__':
    print("begin....")
    for i in range(NUM_WORKERS * 2):
       q.put(i) #放入到任务队列中去
    print("job qsize: %d" %(q.qsize()))

    pool = ThreadPool(8)
    pool.map(doJob, q)
    pool.close()
    pool.join()

    # for x in range(NUM_WORKERS):
    #    MyThread(q,x).start()