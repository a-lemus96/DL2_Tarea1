import os

# Check if files have already been moved. If so, exit the program...
if os.path.exists('left_X') or os.path.exists('right_X') or os.path.exists('target_Y'):
    print("Training files have already been placed into their corresponding folers.")
    exit()

# If not, proceed with program execution
source_target_pairs = [('left_X', 'final_left'), ('right_X', 'final_right'), ('target_Y', 'disparities_viz')]

for target_folder, source_folder in source_target_pairs:
    folders = sorted(os.listdir(f'training/{source_folder}'))
    
    for folder in folders:
        folder_path = f'training/{source_folder}/{folder}'
        print(folder_path)
        target_path = f'training_data/{target_folder}'

        for i, file in enumerate(sorted(os.listdir(folder_path))):
            target_file = f'{target_path}/{folder}_{file.split("_")[1]}'
            source_file = f'{folder_path}/{file}'
            print(f'Moving {source_file} to {target_file} ...')
            os.system(f"mv {source_file} {target_file}")