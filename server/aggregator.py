import torch

def fedavg_aggregate_dp(global_model, client_models, client_sizes,
                        noise_sigma=1.0, clip_norm=1.0):
    import torch

    global_dict = global_model.state_dict()
    total_samples = sum(client_sizes)

    new_state_dict = {}

    for key in global_dict.keys():
        new_state_dict[key] = torch.zeros_like(global_dict[key])

        for i, client_model in enumerate(client_models):
            weight = client_sizes[i] / total_samples
            new_state_dict[key] += weight * client_model.state_dict()[key]

        noise = torch.randn_like(new_state_dict[key]) * noise_sigma * clip_norm
        new_state_dict[key] += noise

    global_model.load_state_dict(new_state_dict)
    return global_model