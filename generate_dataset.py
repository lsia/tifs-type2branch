import  numpy as np;
import  os;
import  pickle;
import  random;
import  shutil;
import  sys;



KVC_DATASET_FOLDER = "C:/Users/nahue/Downloads/KVC_data_for_participants/KVC_data_for_participants/";
# KVC_DATASET_FOLDER = "raw/";



def print_help():
    print("generate_dataset.py {desktop|mobile} {training users}");
    exit(-1);
    

if len(sys.argv) != 3:
    print_help();
if sys.argv[1] != "desktop" and sys.argv[1] != "mobile":
    print_help();

SCENARIO = sys.argv[1];
TRAINING_USERS = int(sys.argv[2]) if sys.argv[2].isdecimal() else None;
if TRAINING_USERS is None or TRAINING_USERS < 0:
    print_help();


print("Generating training/validation/evaluation dataset...");
print("  Scenario:       " + SCENARIO);
print("  Training users: " + str(TRAINING_USERS));


DATASET = "KVC" + SCENARIO + str(TRAINING_USERS);
print("  Dataset name:   " + DATASET);
if not os.path.exists("datasets/"):
    os.mkdir("datasets/");
if not os.path.exists("datasets/" + DATASET + "/"):
    os.mkdir("datasets/" + DATASET + "/");


print("Loading KVC training set: ")

FILE = SCENARIO + "_dev_set.npy";
print("  Filename:  " + FILE);
FULLPATH = KVC_DATASET_FOLDER + SCENARIO + "/" + FILE;
print("  Full path: " + FULLPATH);
print("  Loading...");
data = np.load(FULLPATH, allow_pickle=True).item();

users = list(data.keys());
print("  Total users: " + str(len(users)));
if TRAINING_USERS > (len(users) - 2000):
    print("ERROR!!! There are not enough users in the training set (" + str(TRAINING_USERS) + " training, 1000 validation, 1000 evaluation needed).");
    exit(-1);

print("Splitting users...");
available = list(range(len(users)));

training_users = random.sample(available, TRAINING_USERS);
training_users_names = [users[i] for i in training_users]
print("  Training users (" + str(len(training_users_names)) + "): " + str(training_users_names));

available = [u for u in available if u not in training_users];
validation_users = random.sample(available, 1000);
validation_users_names = [users[i] for i in validation_users]
print("  Validation users (" + str(len(validation_users_names)) + "): " + str(validation_users_names));

available = [u for u in available if u not in validation_users];
evaluation_users = random.sample(available, 1000);
evaluation_users_names = [users[i] for i in evaluation_users]
print("  Evaluation users (" + str(len(evaluation_users_names)) + "): " + str(evaluation_users_names));

FOLDER_NPY = "datasets/" + DATASET + "/npy/";
if not os.path.exists(FOLDER_NPY):
    os.mkdir(FOLDER_NPY);

FOLDER_CSV = "datasets/" + DATASET + "/csv/";
if not os.path.exists(FOLDER_CSV):
    os.mkdir(FOLDER_CSV);


def prepare_sample(sample, MAX_MILLISECONDS):
    CROP_VALUE = 30.0;
    m = np.zeros([100,3]);

    max_pos = len(sample);
    if max_pos > 100:
        max_pos = 100;

    for i in range(0, max_pos):
        dp = sample[i];

        if dp[2] < 0 or dp[2] > 255:
            print("  INVALID ASCII CODE!!!", dp);
            m[i,0] = 0.0;  
        else:
            m[i,0] = dp[2] / 255.0;

        ht = (dp[1] - dp[0]) / MAX_MILLISECONDS;
        if ht < 0.0:
            print("    Negative HT (" + str(ht) + ") at pos " + str(i), dp);
            ht = 0.0;
        elif ht > CROP_VALUE:
            ht = CROP_VALUE;

        m[i,1] = ht;

        if i == 0:
            m[i,2] = 0.0;
        else:
            ft = (dp[0] - sample[i-1][0]) / MAX_MILLISECONDS;
            if ft < 0.0:
                print("    Negative FT (" + str(ft) + ") at pos " + str(i), dp);
                ft = 0.0;
            elif ft > CROP_VALUE:
                ft = CROP_VALUE;

            m[i,2] = ft;

    return m;



def get_best_15(user, samples):
    retval = samples;
    if len(samples) > 15:
        print("USER " + user + " has more than 15 samples (" + str(len(samples)) + "). Choosing best...");

        candidates = [];
        for sample_id, sample in samples.items():
            variety = len(np.unique(sample[:,2])); 

            condition_1 = (sample[:, 2] >= 65) & (sample[:, 2] <= 90)
            condition_2 = (sample[:, 2] >= 48) & (sample[:, 2] <= 57)
            final_condition = condition_1 | condition_2
            alphanumeric_rate = np.count_nonzero(final_condition) / len(sample);

            score = (variety / 20) * alphanumeric_rate * len(sample);

            # print("variety=", variety);
            # print("alphanumeric_rate=", alphanumeric_rate);
            # print("score=", score);
            # print(np.unique(sample[:,2]));
            # input();

            candidates.append((score, sample_id, sample));

        sorted_candidates = sorted(candidates, key=lambda x: x[0], reverse=True);
        top_15 = sorted_candidates[:15];
        
        retval = {};
        for score, sample_id, sample in top_15:
            retval[sample_id] = sample;

    return retval;



def generate_dataset_part(name, user_names):
    print("Generating dataset part " + name + "...");
    if not os.path.exists(FOLDER_CSV + name + "/"):
        os.mkdir(FOLDER_CSV + name + "/");

    x = {};
    n_user = -1;
    for user_name in user_names:
        n_user += 1;
        x[user_name] = {};
        samples = data[user_name];
        print("  USER " + user_name + " (" + str(len(samples)) + " samples)");
        samples = get_best_15(user_name, samples);

        n_sample = -1;
        for sample_id, arr in samples.items():
            n_sample += 1;
            sample = prepare_sample(arr, 1000.0);
            x[user_name][sample_id] = sample;
    
            CSVNAME = FOLDER_CSV + name + "/" + f"U{n_user}S{n_sample}.csv";
            np.savetxt(CSVNAME, sample, delimiter=",", fmt="%s");

    np.save(FOLDER_NPY + name + ".npy", x, allow_pickle=True);

generate_dataset_part("xt", training_users_names);
generate_dataset_part("xv", validation_users_names);
generate_dataset_part("xe", evaluation_users_names);

print("Done.");