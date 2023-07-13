import torch
import subprocess
import numpy as np
import plotly.graph_objs as go
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
    def __init__(self, heldin, recon_data, vel, conds, device):
        super().__init__()
        self.heldin = heldin.to(device)
        self.recon_data = recon_data.to(device)
        self.vel = vel.to(device)
        self.conds = conds.to(device)

    def __len__(self):
        return self.heldin.shape[0]

    def __getitem__(self, i):
        return self.heldin[i], self.recon_data[i], self.vel[i], self.conds[i]

def assert_gpu_runtime():
    error_message = '\n\n⚠️ ERROR: GPU is not detected! ⚠️\n\nPlease change the runtime type and re-run the notebook. To do so:\n\tSelect "Runtime" > "Change runtime type" > "Hardware accelerator" > "GPU"'
    sp_call = subprocess.call('nvidia-smi', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # run `nvidia-smi` to see if we can see the GPU, if not raise AssertionError
    assert  sp_call == 0, f'{error_message}'

# function to plot a 3D scatterplot using plotly
def create_3d_scatterplot(data_tuples, xaxis_title, yaxis_title, zaxis_title, fig_size=(700, 500)):
    data = [go.Scatter3d(x=[v[0]], y=[v[1]], z=[v[2]], mode='markers', name=n) for v, n in data_tuples]
    data.append(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=2, color='black'), name='center (0,0,0)'))
    layout_dict = dict(xaxis=dict(title=xaxis_title, range=[-1, 1]),
                       yaxis=dict(title=yaxis_title, range=[-1, 1]),
                       zaxis=dict(title=zaxis_title, range=[-1, 1]),
                       camera=dict(eye=dict(x=1.3, y=-1.3, z=1.3), center=dict(x=0.065, y=0.0, z=-0.075)),
                       aspectmode='cube')
    layout = go.Layout(scene=layout_dict, margin=dict(l=0,r=0,b=0,t=0), width=fig_size[0], height=fig_size[1])
    return go.Figure(data=data, layout=layout).show(config={'displayModeBar': False})


def add_conds_to_trial_data(trial_data_in, dataset_in):
    cond_fields = ['trial_type', 'trial_version']
    combinations = sorted(dataset_in.trial_info[cond_fields].dropna().set_index(cond_fields).index.unique().tolist())
    combinations = np.array(combinations)
    trial_data = trial_data_in.copy()
    trial_info = dataset_in.trial_info
    trial_nums = trial_info.trial_id.values
    trial_data['trial_cond'] = np.zeros(len(trial_data))
    for i,comb in enumerate(combinations):
        # Need a list of all the trial_ids that match cond
        flag1 = trial_info.trial_type.values == comb[0]
        flag2 = trial_info.trial_version.values == comb[1]
        flag3 = np.logical_and(flag1, flag2)
        trial_flag =np.where(flag3) # a list of indices in trial_info that
        cond_trials = trial_nums[trial_flag]
        trial_data.loc[np.isin(trial_data.trial_id, cond_trials),'trial_cond'] = i
    return trial_data

def get_dataloaders(dataset, device):
    spk_field = "spikes"
    hospk_field="heldout_spikes"
    align_field = "move_onset_time"
    align_window= (-250, 450)
    align_field_fwd= "move_onset_time"
    align_window_fwd= (450,650)

    dataloaders = {}

    for trial_split in ["train", "val"]:
        split_to_mask = lambda x: (dataset.trial_info.split == x) if isinstance(x, str) else x
        trial_mask = split_to_mask(trial_split)
        allow_nans = trial_split != "train"
        trial_data = dataset.make_trial_data(ignored_trials=~trial_mask, allow_nans=allow_nans, align_field = align_field, align_range = align_window)
        trial_data = add_conds_to_trial_data(trial_data, dataset)

        trial_data_fwd = dataset.make_trial_data(ignored_trials=~trial_mask, allow_nans=allow_nans, align_field = align_field_fwd, align_range = align_window_fwd)
        trial_data_fwd = add_conds_to_trial_data(trial_data_fwd, dataset)

        grouped = list(trial_data.groupby('trial_id', sort=False))
        grouped_fwd = list(trial_data_fwd.groupby('trial_id', sort=False))

        heldin = torch.Tensor(np.stack([trial[spk_field].to_numpy() for _, trial in grouped]))
        heldout = torch.Tensor(np.stack([trial[hospk_field].to_numpy() for _, trial in grouped]))
        heldin_fwd = torch.Tensor(np.stack([trial[spk_field].to_numpy() for _, trial in grouped_fwd]))
        heldout_fwd = torch.Tensor(np.stack([trial[hospk_field].to_numpy() for _, trial in grouped_fwd]))

        vel= torch.Tensor(np.stack([trial["hand_vel"].to_numpy() for _, trial in grouped]))
        conds = torch.Tensor(np.stack([trial["trial_cond"].to_numpy() for _, trial in grouped]))

        heldin_full = torch.cat([heldin, heldin_fwd], dim=1)
        heldout_full = torch.cat([heldout, heldout_fwd], dim = 1)
        recon_data = torch.cat([heldin_full, heldout_full], dim =2)

        tensor_dataset = Dataset(heldin, recon_data, vel, conds, device)
        dataloader = DataLoader(tensor_dataset,
                                batch_size = 25,
                                num_workers = 4,
                                shuffle = True)

        dataloaders[trial_split] = dataloader
    
    return dataloaders


def get_submission_inputs(dataset):
    from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors

    train_dict = make_train_input_tensors(dataset=dataset,
                                        dataset_name='mc_maze_small',
                                        trial_split='train', # trial_split=['train', 'val'], for Test phase
                                        save_file=False,
                                        include_forward_pred=True)

    eval_dict = make_eval_input_tensors(dataset=dataset,
                                        dataset_name='mc_maze_small',
                                        trial_split='val', # trial_split='test', for Test phase
                                        save_file=False)

    training_input = torch.Tensor(
        np.concatenate([
            train_dict['train_spikes_heldin'],
            np.zeros(train_dict['train_spikes_heldin_forward'].shape), # zeroed inputs for forecasting
        ], axis=1))

    training_output = torch.Tensor(
        np.concatenate([
            np.concatenate([
                train_dict['train_spikes_heldin'],
                train_dict['train_spikes_heldin_forward'],
            ], axis=1),
            np.concatenate([
                train_dict['train_spikes_heldout'],
                train_dict['train_spikes_heldout_forward'],
            ], axis=1),
        ], axis=2))

    eval_input = torch.Tensor(
        np.concatenate([
            eval_dict['eval_spikes_heldin'],
            np.zeros((
                eval_dict['eval_spikes_heldin'].shape[0],
                train_dict['train_spikes_heldin_forward'].shape[1],
                eval_dict['eval_spikes_heldin'].shape[2]
            )),
        ], axis=1))
    
    tlen = train_dict['train_spikes_heldin'].shape[1]
    num_heldin = train_dict['train_spikes_heldin'].shape[2]
    
    return training_input, training_output, eval_input, tlen, num_heldin
