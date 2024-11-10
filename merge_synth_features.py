import  numpy as np;
import  os;
import  pickle;
import  random;
import  shutil;
import  sys;




def print_help():
    print("merge_synth_features.py {dataset}");
    exit(-1);
    

if len(sys.argv) != 2:
    print_help();
if not os.path.exists("datasets/" + sys.argv[1] + "/"):
    print("ERROR!!! Dataset '" + sys.argv[1] + "' not found.");
    print_help();

DATASET = sys.argv[1];



def merge_folder(name, x):
    print("Merging folder " + name + " with " + name + "s...");
    CSV_HUMAN = "datasets/" + DATASET + "/csv/" + name + "/";
    CSV_SYNTH = "datasets/" + DATASET + "/csv/" + name + "s/";

    xmerged = {};
    n_user = -1;
    for user_id, samples in x.items():
        n_user += 1;
        if n_user % 1000 == 0:
            print(str(n_user) + " USERS");
            
        print(".", end="", flush=True);
        
        xmerged[user_id] = {};
        
        n_sample = -1;
        for sample_id, nparr in samples.items():
            n_sample += 1;

            file = "U" + str(n_user) + "S" + str(n_sample) + ".csv";
            if not os.path.exists(CSV_HUMAN + file):
                print("  ERROR!!! File " + file + " missing from " + CSV_HUMAN);
                exit(-1);
            if not os.path.exists(CSV_SYNTH + file):
                print("  ERROR!!! File " + file + " missing from " + CSV_SYNTH);
                exit(-1);

            csvh = np.genfromtxt(CSV_HUMAN + file, delimiter=',', dtype=float);

            csvs = np.genfromtxt(CSV_SYNTH + file, delimiter=',', dtype=float);
            if len(csvs.shape) == 1:
            	csvs = np.zeros([100,3]);
            	print("X", end="", flush=True);
            else:
            	padding = np.full((100 - csvs.shape[0], csvs.shape[1]), 0.0, dtype=float);
            	csvs = np.vstack((csvs, padding))

            combined = np.hstack((csvh, csvs[:, 1:]))
            xmerged[user_id][sample_id] = combined;
       
    return xmerged;

FOLDER_NPY = "datasets/" + DATASET + "/npy/";

OUTPUT_FOLDER = "datasets/" + DATASET + "merged/";
if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER);

OUTPUT_FOLDER = "datasets/" + DATASET + "merged/npy/";
if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER);


xt = np.load(FOLDER_NPY + "xt.npy", allow_pickle=True).item();
xtm = merge_folder("xt", xt);
np.save(OUTPUT_FOLDER + "xt.npy", xtm, allow_pickle=True);

xv = np.load(FOLDER_NPY + "xv.npy", allow_pickle=True).item();
xvm = merge_folder("xv", xv);
np.save(OUTPUT_FOLDER + "xv.npy", xvm, allow_pickle=True);

xe = np.load(FOLDER_NPY + "xe.npy", allow_pickle=True).item();
xem = merge_folder("xe", xe);
np.save(OUTPUT_FOLDER + "xe.npy", xem, allow_pickle=True);
