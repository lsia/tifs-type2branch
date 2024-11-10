import  tensorflow              as tf;
from    tensorflow.keras.layers import Layer, Flatten, Activation, Permute, Embedding, Concatenate;
from    tensorflow.keras.layers import Permute;

import  conf;



class EmbedAndConcatLayer(Layer):
    def __init__(self, num_categories, embedding_dim, **kwargs):
        super(EmbedAndConcatLayer, self).__init__(**kwargs)
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.embedding_layer = Embedding(input_dim=self.num_categories, output_dim=self.embedding_dim)
        
    def call(self, inputs):
        first_feature = inputs[:, :, 0];
        embedded_first_feature = self.embedding_layer(tf.round(first_feature * 255));        
        remaining_features = inputs[:, :, 1:];
        concatenated_features = Concatenate(axis=-1)([embedded_first_feature, remaining_features]);
        return concatenated_features



class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 name='W')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True,
                                 name='b')
        self.v = self.add_weight(shape=(self.units,),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 name='v')
        super(TemporalAttention, self).build(input_shape)

    def call(self, inputs):
        scores = tf.tanh(tf.matmul(inputs, self.W) + self.b);
        scores = tf.matmul(scores, tf.expand_dims(self.v, axis=-1));
        scores = tf.squeeze(scores, axis=-1);

        attention_weights = tf.nn.softmax(scores, axis=1);
        attended_representation = tf.concat([inputs, tf.expand_dims(attention_weights, axis=-1)], axis=-1);
        return attended_representation;



class SelfAttnSequences(Layer):
    def __init__(self, alength, return_sequences = True):
        self.alength = alength
        self.return_sequences = return_sequences
        super(SelfAttnSequences, self).__init__()

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1',
                                  shape=(self.alength, input_shape[2]),
                                  initializer='random_uniform',
                                  trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(input_shape[1], self.alength),
                                  initializer='random_uniform',
                                  trainable=True)
        super(SelfAttnSequences, self).build(input_shape)

    def call(self, inputs):
        W1, W2 = self.W1[None, :, :], self.W2[None, :, :]
        hidden_states_transposed = Permute(dims=(2, 1))(inputs)
        attention_score = tf.matmul(W1, hidden_states_transposed)
        attention_score = Activation('tanh')(attention_score)
        attention_weights = tf.matmul(W2, attention_score)
        attention_weights = Activation('softmax')(attention_weights)
        embedding_matrix = tf.matmul(attention_weights, inputs)
        if not self.return_sequences:
            embedding_matrix = Flatten()(embedding_matrix)
        return embedding_matrix



class ChannelAttention1D(Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super(ChannelAttention1D, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.filters = input_shape[-1]
        self.fc1 = tf.keras.layers.Dense(self.filters // self.reduction_ratio, activation='relu')
        self.fc2 = tf.keras.layers.Dense(self.filters, activation='sigmoid')

    def call(self, inputs):
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(inputs)
        fc1_out = self.fc1(avg_pool)
        fc2_out = self.fc2(fc1_out)
        attention = tf.keras.layers.Reshape((1, self.filters))(fc2_out)
        scaled_features = tf.keras.layers.Multiply()([inputs, attention])
        return scaled_features

    

def get_model_Type2Branch(descriptor):
  print("Model configuration:");
  print("  WIDTH:   " + str(conf.MODEL_WIDTH));
  print("  FILTERS: " + str(conf.MODEL_FILTERS));
  print("  DROPOUT: " + str(conf.MODEL_DROPOUT));
  print("Compiling model...");

  retval = descriptor;
  retval["name"] = "KVCwinner";
  retval["optimizer"] = tf.keras.optimizers.Adam(0.0001);
  retval["normalized"] = False;
  retval["epochs"] = 600;   
  retval["threshold"] = 0.86; 
  
  li  = tf.keras.layers.Input(shape=(descriptor["SEQUENCE_LENGTH"], descriptor["INPUT_FEATURES"]));
  ke = EmbedAndConcatLayer(256,8)(li);
  bn1 = tf.keras.layers.BatchNormalization()(ke);

  p1_ta = TemporalAttention(units=conf.MODEL_WIDTH)(bn1);
  p1_bgru1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(conf.MODEL_WIDTH, return_sequences=True))(p1_ta);
  p1_bn1 = tf.keras.layers.BatchNormalization()(p1_bgru1);
  p1_d1 = tf.keras.layers.Dropout(conf.MODEL_DROPOUT)(p1_bn1);
  p1_sas = SelfAttnSequences(alength=conf.MODEL_WIDTH)(p1_d1);
  p1_bgru2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(conf.MODEL_WIDTH))(p1_sas);
  p1_bn2 = tf.keras.layers.BatchNormalization()(p1_bgru2);
  p1_d2 = tf.keras.layers.Dropout(conf.MODEL_DROPOUT)(p1_bn2);

  p2_ta = TemporalAttention(units=conf.MODEL_WIDTH)(bn1);
  p2_c1 = tf.keras.layers.Conv1D(filters=conf.MODEL_FILTERS, kernel_size=6, activation="relu")(p2_ta);
  p2_bn1 = tf.keras.layers.BatchNormalization()(p2_c1);
  p2_d1 = tf.keras.layers.Dropout(conf.MODEL_DROPOUT)(p2_bn1);
  p2_a1 = ChannelAttention1D()(p2_d1);
  p2_c2 = tf.keras.layers.Conv1D(filters=2*conf.MODEL_FILTERS, kernel_size=6, activation="relu")(p2_a1);
  p2_bn2 = tf.keras.layers.BatchNormalization()(p2_c2);
  p2_d2 = tf.keras.layers.Dropout(conf.MODEL_DROPOUT)(p2_bn2);
  p2_a2 = ChannelAttention1D()(p2_d2);
  p2_c3 = tf.keras.layers.Conv1D(filters=4*conf.MODEL_FILTERS, kernel_size=6, activation="relu")(p2_a2);
  p2_bn3 = tf.keras.layers.BatchNormalization()(p2_c3);
  p2_d3 = tf.keras.layers.Dropout(conf.MODEL_DROPOUT)(p2_bn3);
  p2_a3 = ChannelAttention1D()(p2_d3);
  p2_g = tf.keras.layers.GlobalAveragePooling1D()(p2_a3);
  p2_d4 = tf.keras.layers.Dropout(conf.MODEL_DROPOUT)(p2_g);

  im = tf.keras.layers.Concatenate()([p1_d2, p2_d4]);
  m_dense = tf.keras.layers.Dense(conf.MODEL_WIDTH, activation="relu")(im);
  m_bn = tf.keras.layers.BatchNormalization()(m_dense);
  m_d = tf.keras.layers.Dropout(conf.MODEL_DROPOUT)(m_bn);
  lo = tf.keras.layers.Dense(256, activation=None)(m_d);
  
  retval["model"] = tf.keras.Model(inputs=li, outputs=lo);
  return retval;
