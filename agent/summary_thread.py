import signal
from queue import Empty

import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import logging

class SummaryThread(mp.Process):
    def __init__(self,
                 name: str,
                 input_queue: mp.Queue,
                 actions: list):
        super(SummaryThread, self).__init__()
        self.i_queue = input_queue
        self.name = name
        self.exit = mp.Event()
        self.actions = actions
        self.dict_hist = dict()
        self.dict_hist_count = dict()

    def run(self):
        print("SummaryThread starting")
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
        self.writer = SummaryWriter(self.name)
        while True and not self.exit.is_set():
            try:
                name, scalar, step = self.i_queue.get(timeout=1)

                # Plot histogram of actions
                if name.split('/')[1] == "actions":
                    # Save only 100 histo action
                    if self.dict_hist_count.get(name, None) is None:
                        self.dict_hist_count[name] = 0
                    else:
                        self.dict_hist_count[name] = self.dict_hist_count[name] + 1
                    if self.dict_hist_count.get(name) % 100 != 0:
                        continue

                    if self.dict_hist.get(name, None) is None:
                        self.dict_hist[name] = scalar
                    else:
                        self.dict_hist[name] = self.dict_hist[name] + scalar
                    hist = self.dict_hist[name]

                    fig, ax = plt.subplots()
                    xticks = [i for i in range(len(hist))]
                    ax.bar(xticks, hist, align='center', ec='black')
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(self.actions)
                    fig.autofmt_xdate()
                    self.writer.add_figure(name, fig, step)

                else:
                    self.writer.add_scalar(name, scalar, step)
            except Empty:
                pass
        print("Exiting SummaryThread")

    def stop(self):
        print("Stop initiated for SummaryThread")
        self.exit.set()
