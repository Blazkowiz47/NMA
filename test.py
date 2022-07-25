import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import mindscope_utilities
import mindscope_utilities.visual_behavior_ophys as ophys

from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

data_storage_directory = "F:\\nma\Project\\dataset\\tmp" # Note: this path must exist on your local drive
cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=data_storage_directory)

exp_ids =  [880375092,848697604,950833329, 948507789, 938003662, 1050406399]


def get_processed_timestamps(experiment,t_before=2.0,t_after=2.0,output_sampling_rate=50) -> None:
    '''
    gives the preprocessed neuronal and behaviour data.
    experiment: actual loaded [ophys_experiment]
    t_before: time before stimulus in seconds
    t_after: time after stimulus in seconds
    output_sampling_rate: sampling rate for output
    Returns:
    metadata: 
        dictionary containing :
        'ophys_experiment_id'
        'ophys_session_id'
        'targeted_structure'
        'imaging_depth'
        'equipment_name'
        'cre_line'
        'mouse_id'
        'sex'
    neuronal_data:
        
    running_data
    '''
    tidy_output = ophys.build_tidy_cell_df(experiment)
    exp_id = experiment.ophys_experiment_id
    stimulus_table_with_change = ophys_experiment.stimulus_presentations.query('is_change')
    stimulus_table_without_change = ophys_experiment.stimulus_presentations.query('is_change == False')
    metadata_keys = [
        'ophys_experiment_id',
        'ophys_session_id',
        'targeted_structure',
        'imaging_depth',
        'equipment_name',
        'cre_line',
        'mouse_id',
        'sex',
    ]
    metadata = {}
    for metadata_key in metadata_keys:
        metadata[metadata_key] = str(experiment.metadata[metadata_key])
    
    cells = tidy_output['cell_specimen_id'].unique()
    print('saving metadata')
    with open(data_storage_directory+f'\\processed\\metadata_{exp_id}.json','w') as f:
        json.dump(metadata,f)
    print('done')
    
    print('building running data with change')
    running_data = mindscope_utilities.event_triggered_response(
        data = experiment.running_speed,
        t = 'timestamps',
        y = 'speed',
        event_times = stimulus_table_with_change['start_time'],
        t_before=t_before,
        t_after=t_after,
        output_sampling_rate = output_sampling_rate,
    )
    print('saving running data')
    running_data.to_csv(data_storage_directory+f'\\processed\\running_data_with_change{exp_id}.csv')
    del running_data
    print('building running data wihtout change')
    running_data = mindscope_utilities.event_triggered_response(
        data = experiment.running_speed,
        t = 'timestamps',
        y = 'speed',
        event_times = stimulus_table_without_change['start_time'],
        t_before=t_before,
        t_after=t_after,
        output_sampling_rate = output_sampling_rate,
    )
    print('saving running data')
    running_data.to_csv(data_storage_directory+f'\\processed\\running_data_without_change{exp_id}.csv')
    del running_data
    del experiment.running_speed

    print('building neuronal data change')
    neuronal_data = []
    for cell_id in tqdm(cells):
        etr = mindscope_utilities.event_triggered_response(
        data = tidy_output.query('cell_specimen_id == @cell_id'),
            t = 'timestamps',
            y = 'dff',
            event_times = stimulus_table_with_change['start_time'],
            t_before=t_before,
            t_after=t_after,
            output_sampling_rate = output_sampling_rate,
        )
        neuronal_data.append(etr)
    neuronal_data = pd.concat(neuronal_data)
    print('saving neuronal_data_with_change data')
    neuronal_data.to_csv(data_storage_directory+f'\\processed\\neuronal_data_with_change_{exp_id}.csv')
    del neuronal_data


    print('building neuronal data no change')
    neuronal_data = []
    for cell_id in tqdm(cells):
        etr = mindscope_utilities.event_triggered_response(
        data = tidy_output.query('cell_specimen_id == @cell_id'),
            t = 'timestamps',
            y = 'dff',
            event_times = stimulus_table_without_change['start_time'],
            t_before=t_before,
            t_after=t_after,
            output_sampling_rate = output_sampling_rate,
        )
        neuronal_data.append(etr)
    neuronal_data = pd.concat(neuronal_data)
    print('saving neuronal_data_without_change data')
    neuronal_data.to_csv(data_storage_directory+f'\\processed\\neuronal_data_without_change_{exp_id}.csv')
    del neuronal_data


    
    

for i,e in enumerate(exp_ids):
    if i == 0:
        continue
    ophys_experiment_id = e
    ophys_experiment = cache.get_behavior_ophys_experiment(ophys_experiment_id)
    get_processed_timestamps(ophys_experiment)
    del ophys_experiment



