import multiprocessing as mp
import queue
import threading


class FeedQueue:
    def __init__(self, queue_size=20):
        self.manager = mp.Manager()
        self.bridge_queue = self.manager.Queue(queue_size)
        self.local_queue = queue.Queue(queue_size)
        self.process = []
        self.thread = threading.Thread(target=self.fetch_queue)
        self.flag = True
        self.alive = mp.Value('b', False)
        #self.time = 0

    def start_queue(self, func, args, process_num=2):
        #self.time = time.time()
        self.alive.value = True
        if isinstance(args, list):
            args = self.split_list(args, process_num)
            for i in range(process_num):
                pro = mp.Process(target=self.in_queue, args=(self.alive, self.bridge_queue, func, args[i],))
                pro.daemon = True
                pro.start()
                self.process.append(pro)
        else:
            for i in range(process_num):
                pro = mp.Process(target=self.in_queue, args=(self.alive, self.bridge_queue, func, args,))
                pro.daemon = True
                pro.start()
                self.process.append(pro)
        
        self.thread.daemon = True
        self.thread.start()

    @staticmethod
    def in_queue(alive, q, func, args):
        mp.freeze_support()
        if isinstance(args, list):
            for arg_now in args:
                if not alive.value:
                    break
                tmp = func(*arg_now)
                while alive.value: 
                    if q.full():
                        continue
                    q.put(tmp)
                    break
                    #print('bridge_queue: %d' % q.qsize())
        else:
            while alive.value:
                tmp = func(*args)
                while alive.value:
                    if q.full():
                        continue
                    q.put(tmp)
                    break
                    #print('bridge_queue: %d' % q.qsize())

    def fetch_queue(self):
        while self.flag:
            if self.bridge_queue.empty():
                continue
            tmp = self.bridge_queue.get()
            while self.flag:
                if self.local_queue.full():
                    continue
                self.local_queue.put(tmp)
                #print('local_queue: %d' % self.local_queue.qsize())
                #if self.local_queue.qsize()==20:
                #    print(time.time()-self.time)
                break

    def feed_me(self):
        return self.local_queue.get()

    def close_queue(self):
        self.flag = False
        self.alive.value = False
        if not self.bridge_queue.empty():
            try:
                self.bridge_queue.get_nowait()
            except queue.Empty:
                pass
        for pro in self.process:
            pro.join(5)

    @staticmethod
    def split_list(raw, num_of_sections):
        split = []
        for i in range(num_of_sections):
            split.append([])
        for i, arg in enumerate(raw):
            split[i%num_of_sections].append(arg)
        return split
