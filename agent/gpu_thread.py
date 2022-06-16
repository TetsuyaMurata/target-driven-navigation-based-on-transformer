import queue

import torch
import torch.multiprocessing as mp

from torchvision import transforms


def preprocess_caffe(x):
    x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    return x


class GPUThread(mp.Process):
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 input_queues: mp.Queue,
                 output_queues: mp.Queue,
                 scenes,
                 h5_file_path,
                 evt):
        super(GPUThread, self).__init__()
        self.model = model.eval()
        self.model = self.model.to(device)
        self.device = device
        self.i_queues = input_queues
        self.o_queues = output_queues
        self.exit = mp.Event()
        self.scenes = scenes
        self.evt = evt
        self.preprocess = transforms.Normalize(mean=[123.68, 116.779, 103.939],
                                               std=[1.0, 1.0, 1.0])

    def run(self):
        self.model = self.model.to(self.device)
        print("GPUThread starting")
        while True and not self.exit.is_set():
            self.evt.wait()
            for ind, i_q in enumerate(self.i_queues):
                try:
                    frame = i_q.get(block=False)
                    tensor = frame.to(self.device)
                    tensor = tensor.permute(2, 0, 1)
                    tensor = self.preprocess(tensor)
                    tensor = tensor.unsqueeze(0)
                    output_tensor = self.model(tensor)
                    output_tensor = output_tensor.cpu()
                    self.o_queues[ind].put(output_tensor)
                    self.evt.clear()

                except queue.Empty as e:
                    pass

    def stop(self):
        print("Stop initiated for GPUThread")
        self.exit.set()
        self.evt.set()
