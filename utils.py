import subprocess
import plotly.graph_objs as go
import numpy as np

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