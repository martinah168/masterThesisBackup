from pathlib import Path

ukbb_base_path = Path('~/../projects/ukbb/brain/data').expanduser().resolve()

iter_dir = ukbb_base_path.iterdir()
n_files_hist = {}
mri_seq_hist = {}

for i_dir, cur_dir in enumerate(iter_dir):
    print(f'i_dir: {i_dir}, cur_dir: {cur_dir}')
    files = list(cur_dir.glob('*'))
    num_files = len(files)
    n_files_hist[num_files] = n_files_hist.get(num_files, 0) + 1

    for fn in files:
        mri_seq = fn.name
        mri_seq_hist[mri_seq] = mri_seq_hist.get(mri_seq, 0) + 1

    print(f'n_files hist: {n_files_hist}')
    print(f'mri_seq hist: {mri_seq_hist}')
