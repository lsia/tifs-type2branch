import  numpy       as np;
import  random;
import  tensorflow  as tf;

import  conf;



def calculate_centroids(model, x, xflattened):
    yp = model.predict(xflattened);

    centroids = {};
    count = -1;
    for user_id in x.keys():
        count += 1;
        
        centroid = np.mean(yp[15 * count: 15 * count + 15, :], axis=0);
        centroids[user_id] = centroid;

    return centroids;



class CurriculumSetsGenerator:
    def __init__(self, descriptor, x, K, delay, max_neighbours):
        self.descriptor = descriptor;
        self.x = x;
        self.K = K;
        self.delay = delay;
        self.max_neighbours = max_neighbours;

        self.epoch = 0;
        self.users = list(x.keys());
        self.kdtree = None;
    
    
    def initialize(self):
        self.user_pos = {};
        all_samples = [];
        
        count = -1;
        for user_id, samples in self.x.items():
            count += 1;
            self.user_pos[user_id] = count;

            for sample_id, sample in samples.items():
                all_samples.append(sample);
    
        self.xflattened = np.stack(all_samples);
        

    def on_epoch_end(self, epoch, logs=None):
        pass;

    def on_epoch_begin(self, epoch, logs=None):
        print("[CurriculumSets] EPOCH " + str(epoch));
        self.epoch = epoch;

        if self.epoch < self.delay:
            print("[CurriculumSets] Naive training without challenge.");
        else:
            print("[CurriculumSets] Calculating user centroids...");
            self.centroids = calculate_centroids(self.descriptor["model"], self.x, self.xflattened);

            indexed_centroids = [];
            for i in range(0, len(self.centroids)):
                indexed_centroids.append(self.centroids[self.users[i]]);

            print("[CurriculumSets] Generating spatial index tree...");
            from scipy.spatial import cKDTree;
            self.tree = cKDTree(indexed_centroids);


    def get_random_sets_batch(self):
        X = [];
        Y = [];

        batch_users = random.sample(self.users, self.K);
        
        i = -1;
        for batch_user in batch_users:
            i += 1;
            samples = self.x[batch_user];
            for sample_id, sample in samples.items():
                X.append(sample);
                Y.append(i);

        return np.stack(X), np.stack(Y);


    def get_sets_nearest(self, legitimate_user, samples, nearest_users):
        BATCH_SIZE = conf.N * conf.K;

        X = [];
        Y = [];
        
        print("[CurriculumSets] USER " + legitimate_user);
        pos = self.user_pos[legitimate_user];
        X.extend(self.xflattened[conf.N * pos:conf.N * pos + conf.N,:]);
        Y.extend([pos] * conf.N);

        print("[CurriculumSets] Nearest users:");
        distances, indices = self.tree.query(self.centroids[legitimate_user], k=nearest_users);

        impostors = [];
        for i in range(1, nearest_users):
            impostor  = self.users[indices[i]];
            impostors.append(impostor);

            d         = distances[i];
            index     = indices[i];
            print("[CurriculumSets]   USER " + impostor + " (" + str(index) + ")   d=" + str(d));

            pos = index;
            X.extend(self.xflattened[conf.N * pos:conf.N * pos + conf.N,:]);
            Y.extend([pos] * conf.N);

        while len(X) < BATCH_SIZE:
            impostor = random.choice(self.users);
            if impostor != legitimate_user and impostor not in impostors:
                impostors.append(impostor);
                
                pos = self.user_pos[impostor];
                X.extend(self.xflattened[conf.N * pos:conf.N * pos + conf.N,:]);
                Y.extend([pos] * conf.N);

        X = np.stack(X);
        Y = np.stack(Y);
        return (X, Y);



    def __call__(self):
        BATCH_SIZE = conf.N * conf.K;

        while True:
            for legitimate_user, samples in self.x.items():
                if self.epoch < self.delay:
                    yield self.get_random_sets_batch();
                else:
                    nearest_users = self.epoch // self.delay;
                    
                    max_nearest_users = (BATCH_SIZE // self.max_neighbours) - 2;
                    if nearest_users >= max_nearest_users:
                        nearest_users = max_nearest_users;

                    if nearest_users >= self.K - 2:
                        nearest_users = self.K - 2;

                    yield self.get_sets_nearest(legitimate_user, samples, nearest_users + 1);




class GeneratorCallback(tf.keras.callbacks.Callback):
    def __init__(self, generator):
        self.generator = generator;

    def on_epoch_begin(self, epoch, logs=None):
        self.generator.on_epoch_begin(epoch, logs);

    def on_epoch_end(self, epoch, logs=None):
        self.generator.on_epoch_end(epoch, logs);


def get_generator(descriptor, x):
    print("    CurriculumSets");
    retval = CurriculumSetsGenerator(descriptor, x, conf.K, conf.CURRICULUM_DELAY, conf.CURRICULUM_MAX_NEIGHBOURS);
    retval.initialize();
    
    tc = GeneratorCallback(retval);
    return retval(), retval, tc;


