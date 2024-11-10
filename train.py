import  conf;
import  matplotlib.pyplot       as plt;
import  numpy                   as np;
import  os;
import  sys;
import  tensorflow              as tf;


import  util;



def print_help():
    print("train.py {dataset}");
    exit(-1);
    

#
# Verify command line parameters
#

if len(sys.argv) != 2:
    print_help();
if not os.path.exists("datasets/" + sys.argv[1] + "/"):
    print("ERROR!!! Dataset '" + sys.argv[1] + "' not found.");
    print_help();

DATASET = sys.argv[1];


#
# Load and verify the dataset
#

FOLDER_NPY = "datasets/" + DATASET + "/npy/";

print("Loading dataset " + DATASET + "...");
print("  Training samples...");
xt = np.load(FOLDER_NPY + "xt.npy", allow_pickle=True).item();
print("  Validation samples...");
xv = np.load(FOLDER_NPY + "xv.npy", allow_pickle=True).item();

print("  Verifying sample shapes...");
first_user = list(xt.keys())[0];
first_sample_id = list(xt[first_user].keys())[0];
first_sample = xt[first_user][first_sample_id];
print("    Expected shape: " + str(first_sample.shape));

SEQUENCE_LENGTH = first_sample.shape[0];
INPUT_FEATURES = first_sample.shape[1];

for user, samples in xt.items():
    for sample_id, arr in samples.items():
        if arr.shape != first_sample.shape:
            print("ERROR!!! All samples must have the same shape (found " + str(arr.shape) + " in xt.npy).");

for user, samples in xv.items():
    for sample_id, arr in samples.items():
        if arr.shape != first_sample.shape:
            print("ERROR!!! All samples must have the same shape (found " + str(arr.shape) + " in xv.npy).");



#
# Initialize the training and validation generators
#

descriptor = {};
descriptor["SEQUENCE_LENGTH"] = SEQUENCE_LENGTH;
descriptor["INPUT_FEATURES"] = INPUT_FEATURES;

print("Initializing generators...");
print("  N (samples per user):        " + str(conf.N));
print("  K (sets per batch):          " + str(conf.K));
print("  Sample length (keystrokes):  " + str(SEQUENCE_LENGTH));
print("  Input features:              " + str(INPUT_FEATURES));

print("  Initializing training generator...");
import  training_generator;
ti, tg, tc = training_generator.get_generator(descriptor, xt);

print("  Initializing validation generator...");
import  validation_generator;
vi, vg, vc = validation_generator.get_generator(descriptor, xv);


#
# Initialize loss function
#

print("Initializing loss function...");
import  loss;
l = loss.get_loss();


#
# Compile the model
#

util.clean_folder("model/");

import  model;
descriptor = model.get_model_Type2Branch(descriptor);
m = descriptor["model"];
m.summary();

m.compile(
    optimizer=descriptor["optimizer"],
    loss=l,
);

#
# Initialize callbacks
#

class SaveModelFromEpochCallback(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, save_start_epoch=10, **kwargs):
        super(SaveModelFromEpochCallback, self).__init__(filepath, **kwargs)
        self.save_start_epoch = save_start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.save_start_epoch:
            super(SaveModelFromEpochCallback, self).on_epoch_end(epoch, logs)



print("Initializing callbacks...");
callbacks = [
    SaveModelFromEpochCallback(
        "model/",
        save_start_epoch        = 1,
        save_best_only          = True,
        save_weights_only       = True,
        initial_value_threshold = conf.EARLY_STOP_THRESHOLD,
        verbose                 = 2),
    tf.keras.callbacks.EarlyStopping(
        min_delta=0.0001,
        patience=conf.EARLY_STOP_PATIENCE),        
];

if tc != None:
    callbacks.append(tc);
if vc != None:
    callbacks.append(vc);


#
# Train the model
#

print("Training model...");
history = descriptor["model"].fit(
    ti,
    validation_data     = vi,
    epochs              = conf.EPOCHS,                    
    steps_per_epoch     = conf.TRAINING_STEPS,
    validation_steps    = conf.VALIDATION_STEPS,
    callbacks           = callbacks,
    verbose             = 2,
);



#
# Save the loss history
#

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.savefig("LOSS.png");
