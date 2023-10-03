import pandas as pd
import os
from preprocessing_module import full_cleaning_pipeline

# Constants & Configuration
METAPROJECT_NAME = 'TopicModelling_META'
SUBPROJECT_NAME = 'TopMod_pipeline'
DATASET_NAME = "SensoryTool_CombinedData.csv"
HIGH_SENSORY = True
CONDITION = 'highsensory' if HIGH_SENSORY else 'deeplistening'

PROJDIR = os.path.expanduser(f"~/projects/{METAPROJECT_NAME}")
DATADIR = os.path.join(PROJDIR, f'DATA/{DATASET_NAME}')
CODEDIR = os.path.join(PROJDIR, f'{SUBPROJECT_NAME}')

df = pd.read_csv(DATADIR)
dataset = df[df['meta_HighSensory'] == HIGH_SENSORY]['reflection_answer']
reports = dataset[dataset.notna() & (dataset != '')].reset_index(drop=True)
reports = pd.DataFrame(reports)

df_clean = full_cleaning_pipeline(reports, 'reflection_answer')

# Save the cleaned data
base_name, ext = os.path.splitext(DATASET_NAME)
new_path = f"{base_name}_{CONDITION}_preprocessed{ext}"
preproc_path = os.path.join(PROJDIR, f'DATA/preprocessed/{new_path}')
df_clean.to_csv(preproc_path, index=False)
