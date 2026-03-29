#test.py
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
x = torch.randn(10000,10000).cuda()
print(x.device)