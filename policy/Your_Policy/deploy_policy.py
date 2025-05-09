# import packages and module here

def encode_obs(observation): # Post-Process Observation
    obs = observation
    # ...
    return obs

def get_model(usr_args): # from your deploy_policy.yml
    return Your_Policy(usr_args) # load your model

def eval(TASK_ENV, model, observation):
    '''
        All the function interfaces below are just examples 
        You can modify them according to your implementation
        But we strongly recommend keeping the code logic unchanged
    '''
    obs = encode_obs(observation) # Post-Process Observation
    instruction = TASK_ENV.get_instruction()

    if len(model.obs_cache) == 0: # Force an update of the observation at the first frame to avoid an empty observation window
        model.update_obs(obs)    

    actions = model.get_action() # Get Action according to observation chunk

    for action in actions: # Execute each step of the action
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs) # Update Observation

def reset_model(model): # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    pass