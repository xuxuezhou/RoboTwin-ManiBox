import os, yaml, sys, json
sys.path.append('./')
# from envs._GLOBAL_CONFIGS import *

names = [
    "blocks_ranking_rgb", 
    "blocks_stack_three",
    "dual_shoes_place", 
    "place_object_scale",
    "put_bottles_dustbin",
    "place_phone_stand"
]
for task_name in names:
    task_config_path = os.path.join(f'./task_config', f'{task_name}.yml')

    with open(os.path.join('./script', '_task_config_template.json'), 'r') as file:
        task_config_template = json.load(file)
    
    task_config_template['task_name'] = task_name
    with open(task_config_path, 'w') as f:
        yaml.dump(task_config_template,f,default_flow_style = False,sort_keys=False)