import torch
import numpy as np
size = 20
torch.manual_seed(1234)
testdata = torch.FloatTensor(np.random.rand(200, size, 5))
testdata = torch.cat((testdata, torch.zeros([200, size, 1])), dim=-1)
torch.save(testdata, 'testdata_tspt2_3o_size{}.pt'.format(size))
