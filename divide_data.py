import os
import random
from tqdm import tqdm
from shutil import copyfile


RATIO=0.8
def divide(DATA_PATH,DST_DATA_PATH):
    src_input_dir = os.path.join(DATA_PATH,'images')
    src_lbl_dir = os.path.join(DATA_PATH,'labels')

    train_dst_inp_dir=os.path.join(DST_DATA_PATH,'train','images')
    train_dst_lbl_dir=os.path.join(DST_DATA_PATH,'train','labels')
    val_dst_inp_dir=os.path.join(DST_DATA_PATH,'val','images')
    val_dst_lbl_dir=os.path.join(DST_DATA_PATH,'val','labels')

    os.makedirs(train_dst_inp_dir,0o777,exist_ok=True)
    os.makedirs(train_dst_lbl_dir,0o777,exist_ok=True)
    os.makedirs(val_dst_inp_dir,0o777,exist_ok=True)
    os.makedirs(val_dst_lbl_dir,0o777,exist_ok=True)

    src_input_folders = os.listdir(src_input_dir)
    src_lbl_folders = os.listdir(src_lbl_dir)

    for lbl_folder in src_lbl_folders:
        src_lbl_files = os.listdir(os.path.join(src_lbl_dir,lbl_folder))
        src_input_files = os.listdir(os.path.join(src_input_dir,lbl_folder))

        src_input_names = [inp.split('.')[0] for inp in src_input_files if inp.split('.')[0]+'.csv' in src_lbl_files]
        random.shuffle(src_input_names)

        size_of_input = len(src_input_names)
        train_size=round(size_of_input*RATIO)

        os.makedirs(os.path.join(train_dst_inp_dir, lbl_folder), 0o777, exist_ok=True)
        os.makedirs(os.path.join(train_dst_lbl_dir, lbl_folder), 0o777, exist_ok=True)
        os.makedirs(os.path.join(val_dst_inp_dir, lbl_folder), 0o777, exist_ok=True)
        os.makedirs(os.path.join(val_dst_lbl_dir, lbl_folder), 0o777, exist_ok=True)

        for idx, name in tqdm(enumerate(src_input_names)):



            if idx <train_size:

                with open(os.path.join(src_lbl_dir,lbl_folder,name+'.csv'), 'r') as csvfile:
                    reader = csvfile.read()
                    lbl = reader.split('\n')[0].split(',')
                    # lbl=row
                    # break
                    # row=reader[0]
                if lbl[-1] not in ["FIT_FAILED", "FIND_FAILED"]:
                    dst_train_inp_path=os.path.join(train_dst_inp_dir,lbl_folder, name+'.npy')
                    dst_train_lbl_path=os.path.join(train_dst_lbl_dir,lbl_folder, name+'.csv')
                    copyfile(os.path.join(src_input_dir,lbl_folder,name+'.npy'), dst_train_inp_path)
                    copyfile(os.path.join(src_lbl_dir,lbl_folder,name+'.csv'), dst_train_lbl_path)
            else:
                with open(os.path.join(src_lbl_dir,lbl_folder,name+'.csv'), 'r') as csvfile:
                    reader = csvfile.read()
                    lbl = reader.split('\n')[0].split(',')
                    # lbl=row
                    # break
                    # row=reader[0]
                if lbl[-1] not in ["FIT_FAILED", "FIND_FAILED"]:
                    dst_val_inp_path = os.path.join(val_dst_inp_dir, lbl_folder, name + '.npy')
                    dst_val_lbl_path = os.path.join(val_dst_lbl_dir, lbl_folder, name + '.csv')


                    copyfile(os.path.join(src_input_dir,lbl_folder,name+'.npy'), dst_val_inp_path)
                    copyfile(os.path.join(src_lbl_dir,lbl_folder,name+'.csv'), dst_val_lbl_path)



if __name__ == '__main__':
    DATA_PATH="/home/doy/dataset/fexpr/dest/listup/"
    DST_DATA_PATH="/home/doy/dataset/fexpr/dest/"
    divide(DATA_PATH,DST_DATA_PATH)

