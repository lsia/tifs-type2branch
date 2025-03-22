import  numpy   as np;
import  random;

import  conf;



class RandomSetsGenerator:
    def __init__(self, descriptor, x, K):
        self.descriptor = descriptor;
        self.x = x;
        self.K = K;
    
        self.users = list(self.x.keys());


    def get_random_sets_batch(self):
        X = [];
        Y = [];

        batch_users = random.sample(self.users, self.K);
        
        i = -1;
        for batch_user in batch_users:
            i += 1;
            samples = self.x[batch_user];
            
            count = 0;
            for sample_id, sample in samples.items():
                X.append(sample);
                Y.append(i);
                
                count += 1;
                if count > conf.N:
                    break;

        return np.stack(X), np.stack(Y);


    def __call__(self):
        while True:
            yield self.get_random_sets_batch();




def get_generator(descriptor, x):
    print("    RandomSets");
    retval = RandomSetsGenerator(descriptor, x, conf.K);
    return retval(), retval, None;



