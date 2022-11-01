import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

EMOTIONS = ['Neutral', 'Happy', 'Sad', 'Angry', 'Surprised', 'Scared', 'Disgusted']

def parseVarison(filepath):
    fn = Path(filepath)
    with open(fn) as f:
        lines = f.readlines()
    if len(lines) <= 0:
        print(f'Fail to read {filepath}')
        return None
    lines = [x.strip().split('\t') for x in lines]
    start_time = datetime.strptime(lines[4][1], '%m/%d/%Y %H:%M:%S.%f')
    camid = fn.stem.split('_')[-1]
    frame_rate = float(lines[6][1])
    cols = lines[8]
    validlines = lines[9:]
    if len(validlines) == 0:
        print('No valid lines in', fn.name)
        return None
    try:
        df = pd.DataFrame(np.array(validlines), columns=cols)
    except Exception as e:
        print("Unexpected error in creating dataframe:", e)
        return None
    df = df.replace('FIND_FAILED', np.nan)
    df = df.replace('FIT_FAILED', np.nan)
    # convert emotion entries from str to np.float64. Video Time keeps str type
    df = pd.concat([df[['Video Time']], df[cols[1:]].astype(float)], ignore_index=True, axis=1)
    df = df.rename(columns={x: cols[x] for x in range(len(cols))})
    return df, camid

def parsePeach(filepath, emotions=EMOTIONS):
    fn = Path(filepath)
    with open(fn) as f:
        lines = f.readlines()
    if len(lines) <= 0:
        print(f'Fail to read {filepath}')
        return None
    lines = [x.strip().split('\t') for x in lines]
    camid = Path(lines[5][1]).stem.split('_')[-1]
    frame_rate = float(lines[6][1])
    cols = ['Video Time'] + emotions
    validlines = lines[9:]
    if len(validlines) == 0:
        print('No valid lines in', fn.name)
        return None
    try:
        df = pd.DataFrame(np.array(validlines), columns=cols)
    except Exception as e:
        print("Unexpected error in creating dataframe:", e)
        return None
    df = df.replace('FIND_FAILED', np.nan)
    df = df.replace('FIT_FAILED', np.nan)
    # convert emotion entries from str to np.float64. Video Time keeps str type
    df = pd.concat([df[['Video Time']], df[cols[1:]].astype(float)], ignore_index=True, axis=1)
    df = df.rename(columns={x: cols[x] for x in range(len(cols))})
    return df, camid

def canonical_framerate(orgdf, fr=30.3, columns=EMOTIONS):
    """
    Resample data in framerate fr.
    Ignore framerate of orgdf even if its framerate is fr for fair comparison.
    """
    df = orgdf.copy()
    df['datetime'] = pd.to_datetime(df['Video Time'], format='%H:%M:%S.%f')
    df = df.set_index('datetime')
    df = df[columns].resample('{:.3f}L'.format(1000/fr)).bfill(limit=1).interpolate(limit=1)
    df = df.reset_index()
    df['Video Time'] = df['datetime'].dt.strftime('%H:%M:%S.%f').str[:-3]
    df = df.drop(columns='datetime')[['Video Time'] + columns]
    df[columns] = df[columns].div(df[columns].sum(axis=1), axis=0) # normalize each row
    return df

def readPair(frfile, p2file):
    """
    read and parse a facereader file and a peach file
    convert framerate to canonical 30.3 fps. forced to convert original fps is 30.3
    normalize each row to sum up 1.0 (facereader output is not normalized)
    
    """
    p2df, org_camid_from_p2 = parsePeach(p2file)
    p2df_norm = canonical_framerate(p2df)
    frdf, org_camid = parseVarison(frfile)
    frdf_norm = canonical_framerate(frdf)
    if org_camid != org_camid_from_p2:
        print('Filename in Peach output is not identical to facereader camid', org_camid, org_camid_from_p2)
    return {'p2': {'org': p2df, 'resampled': p2df_norm,},
            'fr': {'org': frdf, 'resampled': frdf_norm,},}
    
def compare(orgdf, p2df, emotions=EMOTIONS):
    compare_results = {}
    for emo in emotions:
        res = np.abs(orgdf[emo] - p2df[emo])
        stats = res.describe() # stats excluding Nans
        compare_results[emo] = {
            'count': stats['count'],
            'mean': stats['mean'],
            'std': stats['std'],
            'max': stats['max'],
            'min': stats['min'],
        }
    return compare_results

def compareFile(frfile, p2file):
    pair = readPair(frfile, p2file)
    return compare(pair['fr']['resampled'], pair['p2']['resampled'])

if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print(f"Usage: python {sys.argv[0]} facereader_file peach_file")
    #     exit()
    #     # sample files to test
    #     #facereaderfile = Path("drryu1_20-117_20220319_BGT_76cb68.txt")
    #     #peachfile = Path("test_20-117_20220319_BGT_camera1.txt")
    # else:
    #     facereaderfile = Path(sys.argv[1])
    #     peachfile = Path(sys.argv[2])

    facereaderfile="./face_reader/drryu1_22-082_20220419_HTP_25d642.txt"
    peachfile="./p2ach/drryu1_22-082_20220419_HTP_25d642.txt"


    max_mean = 0.05
    max_std = 0.1
    diffstats = compareFile(facereaderfile, peachfile)
    for emo in EMOTIONS:
        mean_of_diff = diffstats[emo]['mean']
        std_of_diff = diffstats[emo]['std']
        success = 'PASS' if mean_of_diff < max_mean and std_of_diff < max_std else 'FAIL'
        print(f"{success}: {emo:10} Mean:{mean_of_diff:.6f}   STD:{std_of_diff:.6f}")
