This project uses Stable Baselines3 to train reinforcement learning agents in a custom RIS environment. The training operates in two modes: PHASE and ASSOCIATION.


Association Training (Configured to work with API)

Association training mode receives a dummy phase shift model "phase_model.zip" (or ideally a properly trained one, once the phase shift training is set up). The environment is initialized with multiple receivers. Environemnt is "reset" every 95 steps to stay withing bounds (aka reset-config and rest-association-matrix are run). The performance or "reward" of the model is calculated by adding the coresponding data rates of the rx-tx pairs defined by the association matrix for each tx. (e.g.if tx1: [rx1, rx3], tx2: [rx2, rx4], the overall metric for the performance of the model woutld be [tx1-rx1 rate + tx1-rx3 rate + tx2-rx2 rate + tx2-rx4 rate] )


Phase Training (Changes required to use with API)

In phase mode, the environment is initialized with a single receiver. Model is trained to create configurations for RIS elements depending on receiver position. Changes are required (in env_gym.py, in functions step_phase, reset_phase, comments have been left for the details) depending on how the API is structured for the RIS phase shifts.


To use:

Set the following parameters in env_gym_training.py before training:

    training_mode: either "phase" or "association"

    num_receivers: number of receivers in scene (only relevant for association training)

    ris_dims: dimensions of the RIS (e.g., [4, 4]),

    abs_receiver_position_bounds: bounds for receiver placement (e.g., [200, 200]), 
    used for observation normalization, doesn't need to be exact.

run env_gym_training.py


Notes:  Receiver height is assumed to be constant.
        To train an association model, the used phase model must have
        been trained with the same ris dimensions and receiver bounds.
        The ip for the api is specified in env_gym.
        In both modes, at the end of the training model is saved
        in "phase_model.zip" or "association_model.zip" and can be used for inference




   



