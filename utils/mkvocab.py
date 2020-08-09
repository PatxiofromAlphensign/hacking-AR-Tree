import numpy
from vocab import Vocab
stuff = 'you okay i am good, really i am god'.split()

class id_tf:
    def __init__(self, vocab_dict):
        self.dict = vocab_dict
    
    def get(self,id_, x):
        
        return self.dict[id_]

def testtrain(tokens, test_tokens):
    vocab_dict = {k:v for k,v in enumerate(tokens)}
    id_t = id_tf(vocab_dict)
    vector = []
    for i in range(0,len(vocab_dict.keys()),2):
        vector.append(list(vocab_dict.keys())[i])

    vcab = Vocab(vocab_dict,
             add_pad=True, add_unk=True,
             id_tf=id_t
             )
    idx = []
    dy = []
    for i,word in enumerate(test_tokens):
        idx.append(vcab.word_to_id(word))
    return idx

print(testtrain(stuff, 'fuck you okay'.split()))


