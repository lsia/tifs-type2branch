import  os;
import  tensorflow_addons   as tfa;

import  sys;
import  tensorflow                  as      tf;
from    tensorflow.keras.losses     import  Loss;
from    tensorflow_addons.losses    import  metric_learning;

import  conf;





@tf.function
def calculate_set2set_loss(y_true, y_pred, K, p, beta) -> tf.Tensor:
    labels = tf.convert_to_tensor(y_true, name="labels")
    embeddings = tf.convert_to_tensor(y_pred, name="embeddings")

    convert_to_float32 = (
        embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
    );
    precise_embeddings = (
        tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
    );

    pdist_matrix = metric_learning.pairwise_distance(
        precise_embeddings, squared=False
    );

    retval = 0.0;
    total_pairs = 0;
    for i in range(0, K - 1):
        for j in range(i + 1, K):
            posA = conf.N * i;
            posB = conf.N * j;

            eA = tf.expand_dims(pdist_matrix[posA:posA + conf.N,posA: posA + conf.N],axis=-1);
            eA = tf.tile(eA, [1, 1, conf.N]);

            eB = tf.expand_dims(pdist_matrix[posA:posA + conf.N:,posB:posB + conf.N],axis=1);
            eB = tf.tile(eB, [1, conf.N, 1]);

            m = eA - eB + 1.5;
            m = tf.maximum(m, 0);

            m = tf.multiply(m,p);
            l = tf.reduce_sum(m);
                
            retval += l;
            total_pairs += 1;
      
    retval /= conf.N * conf.N * (conf.N - 1) / 2;
    retval /= total_pairs;
    
    total_radius = 0.0;
    for i in range(0, K):
        legitimate_embeddings = precise_embeddings[conf.N * i:conf.N * i + conf.N];
        centroid = tf.reduce_mean(legitimate_embeddings, axis=0);
        distances = tf.norm(legitimate_embeddings - centroid, axis=1)
        mean_distance = tf.reduce_mean(distances);
        total_radius += mean_distance;

    total_radius /= K;
    total_penalty = 0.0;
    for i in range(0, K):
        legitimate_embeddings = precise_embeddings[conf.N * i:conf.N * i + conf.N];
        centroid = tf.reduce_mean(legitimate_embeddings, axis=0);
        distances = tf.norm(legitimate_embeddings - centroid, axis=1)
        mean_distance = tf.reduce_mean(distances);
        total_penalty += tf.math.abs(mean_distance / total_radius - 1.0);
    
    total_penalty /= K;
    total_penalty *= beta;

    tf.print(tf.strings.format("[Set2Set]    K={}    beta={}    Lsm={}    Lrp={}", [K, beta, retval, total_penalty]), output_stream=sys.stdout);
    return retval + total_penalty;



class Set2SetLoss(Loss):
    def __init__(self, K, beta):
        super(Set2SetLoss, self).__init__()
        self.K = K;
        self.beta = beta;

        p = tf.zeros([conf.N, conf.N, conf.N], dtype=float)

        for i in range(0,conf.N):
          for j in range(i + 1,conf.N):
            for k in range(0,conf.N):
              position = [i, j, k]
              new_value = 1.0
              index = tf.constant([position])
              update = tf.constant([new_value])
              p = tf.tensor_scatter_nd_update(p, index, update);

        self.p = p;


    def call(self, y_true, y_pred):
        return calculate_set2set_loss(y_true, y_pred, self.K, self.p, self.beta);



def get_loss():
    print("    Set2Set (beta=" + str(conf.BETA) + ")");
    return Set2SetLoss(conf.K, conf.BETA);

