import sys
import os

sys.path.append(os.path.abspath("./src/"))

from load import *

def main():
    src_dir = "/data1/aistraj/rest/json"
    src_files = ls_files(src_dir)
    src_abspaths = [os.path.join(src_dir, f) for f in src_files]

    trg_dir = "/data1/aistraj/rest/pickle"
    if not os.path.exists(trg_dir):
        os.mkdir(trg_dir)
    trg_files = [os.path.join(
        trg_dir,
        os.path.basename(file).split('.')[0]+".pickle"
        )
        for file in src_files]

    print("load original json files")
    original_data = load_multiple_trajectoryCollection_parallel(
        src_abspaths,
        num_cores=16
        )
    
    print("dump pickle files to disk")
    for data, file in zip(original_data, trg_files):
        
        pickle_multiple_trajectoryCollection(tcs=data,
                                             file=file
                                             )
    
if __name__ == "__main__":
    sys.exit(main())