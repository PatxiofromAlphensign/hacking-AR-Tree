from basic import NaryLSTMLayer
import torch
import numpy as np
from random import randint


def create_forward_inputs_dummy():
    samed_ones = iterative_forward(10, 4)
    Dx, Dy = np.array(samed_ones.shape)
    h, c = np.array([Dx, 10]), np.array([Dy, 10])
    l = torch.Tensor((h,c*1))
    r = torch.Tensor((h,c*2))
    x = torch.Tensor((h, c*3))
    return l,r,x


def apporach_weight_forward(scale):
    """
    playful mess 

    logic : the weight matrix with a padded torch.ones matrix. and get 
    a random row (i) and insert (i)th column into the ones. And run it as long as
    it does not come on the top

    """
    with torch.enable_grad():
        LSTMscale = NaryLSTMLayer(scale)
        t1 = LSTMscale.comp_linear.weight    
        samed_ones = torch.ones(t1.shape)
        rnd_row = randint(0, t1.shape[0] -1)
        init_sum = sum(samed_ones[0])
       
        while sum(samed_ones[0]) ==  init_sum:
             
            rnd_row = randint(0, t1.shape[0] -1)
            samed_ones[rnd_row] = t1[rnd_row]
            if sum(samed_ones[0])  != init_sum:
                return samed_ones[0]
            #print(np.linalg.norm(forward[0].detach().numpy()))

def iterative_forward(scale, rng):
    
    """
    same alogirthm as apporach_weight_forward
    iterative apporach of modifying samed_ones

    """ 
    with torch.enable_grad():
        LSTMscale = NaryLSTMLayer(scale)
        t1 = LSTMscale.comp_linear.weight    
        samed_ones = torch.ones(t1.shape)
        rnd_row = randint(0, t1.shape[0] -1)
        for _ in range(rng):
            rnd_row = randint(0, t1.shape[0] -1)
            samed_ones[rnd_row] = t1[rnd_row]
        
        return samed_ones

n = NaryLSTMLayer(4)
l,r,x =  create_forward_inputs_dummy()
cat = torch.cat((l,r,x), axis=-2)
cat_pad = torch.zeros(12,20).T
cat_pad[:cat.shape[0], :cat.shape[1]] = cat
n.comp_linear(cat_pad)
print(cat_pad)

#print(n.forward(l,r,x))


#print(apporach_weight_forward(10))
