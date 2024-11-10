import  numpy   as np;
import  os;
import  random;
import  sys;


IMPOSTORS_PER_USER  = 200;



def print_help():
    print("evaluate.py {dataset}");
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
print("  Evaluation samples...");
xe = np.load(FOLDER_NPY + "xt.npy", allow_pickle=True).item();

print("  Verifying sample shapes...");
first_user = list(xe.keys())[0];
first_sample_id = list(xe[first_user].keys())[0];
first_sample = xe[first_user][first_sample_id];
print("    Expected shape: " + str(first_sample.shape));

SEQUENCE_LENGTH = first_sample.shape[0];
INPUT_FEATURES = first_sample.shape[1];

for user, samples in xe.items():
    for sample_id, arr in samples.items():
        if arr.shape != first_sample.shape:
            print("ERROR!!! All samples must have the same shape (found " + str(arr.shape) + " in xt.npy).");


#
# Load the model
#

descriptor = {};
descriptor["SEQUENCE_LENGTH"] = SEQUENCE_LENGTH;
descriptor["INPUT_FEATURES"] = INPUT_FEATURES;

import  model;
descriptor = model.get_model_Type2Branch(descriptor);
m = descriptor["model"];
m.summary();

print("LOAD");
m.load_weights("model/");



#
# Calculate embeddings
#

print("Calculating embeddings...");

embeddings_by_user = {};
for user_id, samples in xe.items():
    print(".", end="", flush=True);
    user_samples = [];
    for sample_id, sample in samples.items():
        user_samples.append(sample);

    user_samples = np.stack(user_samples);
    yl = m.predict(user_samples, verbose=0);
    embeddings_by_user[user_id] = yl;

print("");



#
# Evaluate authentication results
#

def calculate_score(gallery_samples, query_sample):
    s = 0.0;
    for gallery_sample in gallery_samples:
        d = np.linalg.norm(gallery_sample - query_sample);
        s += d;

    s /= len(gallery_samples);
    return s;



def calculate_eer(legitimate_scores, impostor_scores):
    ITERATIONS = 20;

    mind = min(legitimate_scores[0], impostor_scores[0]);
    maxd = max(legitimate_scores[-1], impostor_scores[-1]);

    threshold = (mind + maxd) / 2.0;
    for i in range(0,ITERATIONS):
        threshold = (mind + maxd) / 2.0;
    
        frr_count = len(legitimate_scores) - np.searchsorted(legitimate_scores, threshold, side = "left");
        far_count = np.searchsorted(impostor_scores, threshold, side = "left");

        frr = 100.0 * frr_count / len(legitimate_scores);
        far = 100.0 * far_count / len(impostor_scores);

        if frr == far:
            break;
        elif frr > far:
            mind = threshold;
        else:
            maxd = threshold;

    return (threshold, (far + frr)/2);


G = [1,2,5,7,10];

eers = {};
cumulative_sum   = {};
cumulative_count = {};
global_legitimate_scores = {};
global_impostor_scores = {};

for g in G:
    eers[g] = [];
    cumulative_count[g] = 0.0;
    cumulative_sum[g] = 0.0;
    global_legitimate_scores[g] = [];
    global_impostor_scores[g] = [];

count = -1;
for user_id, yl in embeddings_by_user.items():
    count += 1;
    print("USER " + str(user_id) + " (" + str(count) + "/" + str(len(embeddings_by_user)) + ")");

    yi = [];
    impostor_candidates = [candidate for candidate in embeddings_by_user.keys() if candidate != user_id];
    impostors = random.sample(impostor_candidates, IMPOSTORS_PER_USER);
    for impostor in impostors:
        yi.append(random.choice(embeddings_by_user[impostor]));

    for g in G:
        np.random.shuffle(yl);
        gallery_samples = yl[0:g];
        query_samples = yl[g:];

        legitimate_scores = [];
        for query_sample in query_samples:
            score = calculate_score(gallery_samples, query_sample);
            legitimate_scores.append(score);

        global_legitimate_scores[g].extend(legitimate_scores);
        legitimate_scores.sort();

        impostor_scores = [];
        for query_sample in yi:
            score = calculate_score(gallery_samples, query_sample);
            impostor_scores.append(score);

        global_impostor_scores[g].extend(impostor_scores);
        impostor_scores.sort();

        threshold, eer = calculate_eer(legitimate_scores, impostor_scores);
        eers[g].append(eer);
    
        cumulative_sum[g] += eer;
        cumulative_count[g] += 1.0;
        
        print("  G=" + str(g) + "    EER=" + str(eer) + "   AT " + str(threshold) + "          GLOBAL=" + str(cumulative_sum[g] / cumulative_count[g]));

print("----- GLOBAL AUTHENTICATION RESULTS");
for g in G:
    global_legitimate_scores[g].sort();
    global_impostor_scores[g].sort();
    threshold, eer = calculate_eer(global_legitimate_scores[g], global_impostor_scores[g]);
    print("G=" + str(g) + "    EER=" + str(eer) + "  AT " + str(threshold));

print("----- AVERAGED PER USER AUTHENTICATION RESULTS");
for g in G:
    eer = cumulative_sum[g] / cumulative_count[g];
    print("G=" + str(g) + "    EER=" + str(eer));
