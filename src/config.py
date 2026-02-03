import torch
from multiprocessing import cpu_count

class Config:
    def __init__(self, device="cuda:0", is_half=True):
        self.device = device
        self.is_half = is_half
        self.n_cpu = cpu_count()
        # Inference specific settings from pipeline.py
        self.x_pad = 3
        self.x_query = 10
        self.x_center = 60
        self.x_max = 65
        
        if not is_half:
            self.x_pad = 1
            self.x_query = 6
            self.x_center = 38
            self.x_max = 41
            
        if torch.cuda.is_available():
             gpu_mem = int(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024 + 0.4)
             if gpu_mem <= 4:
                 self.x_pad = 1
                 self.x_query = 5
                 self.x_center = 30
                 self.x_max = 32
