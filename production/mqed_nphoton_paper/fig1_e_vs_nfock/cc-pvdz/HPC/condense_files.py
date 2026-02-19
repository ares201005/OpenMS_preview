#!/usr/bin/env python3
import os
import re
import h5py
import numpy as np

import os
import re
import h5py
import numpy as np

def copy_group(src_group, dest_group):
    """
    Recursively copy all datasets and subgroups from src_group to dest_group.
    """
    for name, item in src_group.items():
        if isinstance(item, h5py.Dataset):
            dest_group.create_dataset(name, data=item[()])
        elif isinstance(item, h5py.Group):
            sub_dest_group = dest_group.create_group(name)
            copy_group(item, sub_dest_group)

def consolidate_full_groups(input_dir='VTQED', output_file='VTQED/vtqed_e_vs_nfock.h5'):
    files = []
    if input_dir == "SCQED":
        pattern = re.compile(r'cc-pvdz_scqed_e_vs_nfock_(\d+\.\d+)\.h5$')
    elif input_dir == "VTQED":
        pattern = re.compile(r'cc-pvdz_vtqed_e_vs_nfock_(\d+\.\d+)\.h5$')

    for filename in os.listdir(input_dir):
        match = pattern.match(filename)
        if match:
            lambda_val = float(match.group(1))
            files.append((filename, lambda_val))

    files.sort(key=lambda x: x[1])  # Sort by lambda

    with h5py.File(output_file, 'w') as out_f:

        for filename, lambda_val in files:
            filepath = os.path.join(input_dir, filename)
            lambda_str = f"{lambda_val:.3f}"
            group_name = f"lambda_{lambda_str}"

            with h5py.File(filepath, 'r') as in_f:
                if group_name in in_f:
                    in_group = in_f[group_name]
                    out_group = out_f.create_group(group_name)
                    copy_group(in_group, out_group)
                else:
                    print(f"Warning: {group_name} not found in {filename}")

if __name__ == "__main__":
    consolidate_full_groups(input_dir="SCQED", output_file="SCQED/scqed_e_vs_nfock.h5")
    consolidate_full_groups(input_dir="VTQED", output_file="VTQED/vtqed_e_vs_nfock.h5")
