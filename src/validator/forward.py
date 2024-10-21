# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import bittensor as bt
import re
import onnx_tool
import torch.onnx
from src.protocol import Dummy
from src.validator.reward import get_rewards
from src.utils.uids import get_random_uids
import torch
from datetime import datetime
import pandas as pd
import requests
import bittensor as bt
import asyncio
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
from model.storage.disk import utils
from model.vali_trainer import ValiTrainer
from model.model_analysis import ModelAnalysis
from model.vali_config import ValidationConfig
import traceback
import plotly.graph_objects as go
import wandb
import os
from torch.profiler import profile, record_function, ProfilerActivity
import math
import numpy as np
from requests.exceptions import ReadTimeout  
import gc
import onnx

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def validate_model_2(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create input tensor
    input_tensor = torch.randn(1, 3, 32, 32).to(device)
    
    # Forward pass
    output = model(input_tensor)

    # Backward pass to compute gradients
    if isinstance(output, tuple):
        output = output[0]  # Take the first 
    output.sum().backward()

    # Force CUDA synchronization
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Inspect the gradients of the model parameters and count tensors with size >= 2
    named_param_tensors_with_grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.dim() >= 2:
            named_param_tensors_with_grads[name] = param.size()

    frozen_count = 0
    grad_count = 0

    excluded_size = torch.Size([1, 3, 32, 32])  # Exclude this size from the frozen count

    # Force garbage collection before iterating
    gc.collect()

    # Iterate through all objects in the garbage collector
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor) and obj.dim() >= 2:
            if obj.requires_grad:
                if obj.grad is not None:
                    print(f"---Detected Tensor with Gradient: {obj.size()}")
                    grad_count += 1
            else:
                # Exclude the specific size from the frozen count
                if obj.size() != excluded_size:
                    print(f"---Detected Frozen Tensor (requires_grad=False): {obj.size()}")
                    frozen_count += 1

    # Only print if the number of frozen tensors is greater
    if frozen_count > grad_count:
        print(f"\n############ Mismatch: Frozen tensor count ({frozen_count}) is larger than gradient tensor count ({grad_count}).")
    


def validate_model_tensors(model):

    input_tensor = torch.randn(1, 3, 32, 32).cuda()
    onnx_path = "tmp.onnx"
    torch.onnx.export(
        model,
        input_tensor,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None  
    )

    # Load the ONNX model
    onnx_model = onnx.load(onnx_path)

    for node in onnx_model.graph.node:
        if node.op_type == 'Scatter':
            print(f"Found Scatter node: {node.name}")
            raise RuntimeError(
            f"Found Scatter node: {node.name}"
                )
        if node.op_type == 'ScatterND':
            print(f"Found ScatterND node: {node.name}")
            raise RuntimeError(
            f"Found ScatterND node: {node.name}"
                )
        if node.op_type == 'ScatterElements':
            print(f"Found ScatterElements node: {node.name}")
            raise RuntimeError(
            f"Found ScatterElements node: {node.name}"
                )
        if node.op_type == 'ConstantOfShape':
            print(f"ConstantOfShape node: {node.name}")
            raise RuntimeError(
            f"Found ConstantOfShape node: {node.name}"
                )

    # Collect tensors with dimension >= 2 from the ONNX model
    onnx_tensors_with_dims = {}
    for initializer in onnx_model.graph.initializer:
        dims = tuple(initializer.dims)
        if len(dims) >= 2:
            # print(f"ONNX Tensor: {initializer.name}, Dimensions: {dims}")
            onnx_tensors_with_dims[initializer.name] = dims

    tensor_count_onnx = len(onnx_tensors_with_dims)

    # Step 2: Forward and backward pass on the PyTorch model
    # Forward pass
    output = model(input_tensor)

    # Backward pass to compute gradients
    if isinstance(output, tuple):
        output = output[0]  # Take the first element if output is a tuple
    output.sum().backward()

    # Collect tensors from named parameters with requires_grad=True and dimension >= 2
    named_param_tensors_with_grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.dim() >= 2:
            named_param_tensors_with_grads[name] = tuple(param.size())

    # Step 3: Collect tensors from garbage collector with gradients and dimension >= 2
    gc_tensors_with_grads = []
    for obj in gc.get_objects():
        if (
            isinstance(obj, torch.Tensor) and
            obj.grad is not None and
            obj.dim() >= 2
        ):
            # print(f"Detected Tensor with Gradient from GC: {obj.size()}")
            gc_tensors_with_grads.append(obj)

    # Step 4: Collect tensors from state_dict with dimension >= 2
    state_dict_tensors_with_dims = {}
    for name, tensor in model.state_dict().items():
        if tensor.dim() >= 2:
            state_dict_tensors_with_dims[name] = tuple(tensor.size())

    # Step 5: Convert collected tensors to sets of sizes for comparison
    named_param_tensor_sizes = set(named_param_tensors_with_grads.values())
    gc_tensor_sizes = {tuple(tensor.size()) for tensor in gc_tensors_with_grads}
    state_dict_tensor_sizes = set(state_dict_tensors_with_dims.values())
    onnx_tensor_sizes = set(onnx_tensors_with_dims.values())

    # Step 6: Compare tensor counts
    named_param_tensor_count = len(named_param_tensor_sizes)
    gc_tensor_count = len(gc_tensor_sizes)
    state_dict_tensor_count = len(state_dict_tensor_sizes)
    onnx_tensor_count = len(onnx_tensor_sizes)

    if (
        named_param_tensor_count != gc_tensor_count or
        named_param_tensor_count != state_dict_tensor_count or
        named_param_tensor_count != onnx_tensor_count
    ):
        raise RuntimeError(
            f"Mismatch in tensor counts!\n"
            f"Named Parameters: {named_param_tensor_count}\n"
            f"Tensors with Gradients (GC): {gc_tensor_count}\n"
            f"State Dict Tensors: {state_dict_tensor_count}\n"
            f"ONNX Tensors: {onnx_tensor_count}"
        )

    # Step 7: Compare tensor sizes between ONNX and state_dict
    missing_in_state_dict = onnx_tensor_sizes - state_dict_tensor_sizes
    missing_in_onnx = state_dict_tensor_sizes - onnx_tensor_sizes

    if missing_in_state_dict or missing_in_onnx:
        raise RuntimeError(
            f"Mismatch in tensor sizes between ONNX and state_dict tensors.\n"
            f"Missing in state_dict: {missing_in_state_dict}\n"
            f"Missing in ONNX: {missing_in_onnx}"
        )

    # Step 8: Compare tensor sizes between named parameters and state_dict
    missing_in_named_params = state_dict_tensor_sizes - named_param_tensor_sizes
    if missing_in_named_params:
        raise RuntimeError(
            f"Tensors in state_dict not found in named parameters: {missing_in_named_params}"
        )

    # Step 9: Compare tensor sizes between GC tensors and named parameters
    missing_in_gc = named_param_tensor_sizes - gc_tensor_sizes
    if missing_in_gc:
        raise RuntimeError(
            f"Tensors in named parameters not found in GC tensors: {missing_in_gc}"
        )



def plot_pareto_after(df, pareto_optimal_points_after):
    fig = go.Figure()

    # Plot all points
    fig.add_trace(go.Scatter(
        x=df['params'],
        y=df['accuracy'],
        mode='markers',
        name='All Points',
        text=df.apply(lambda row: f"UID: {row['uid']}<br>FLOPs: {row['flops']}", axis=1),  # Add UID and FLOPs to hover data
        hovertemplate='%{text}<br>Params: %{x}<br>Accuracy: %{y}<extra></extra>'
    ))

    # Sort and plot Pareto optimal points after validation
    pareto_optimal_points_after = pareto_optimal_points_after.sort_values(by='params')
    fig.add_trace(go.Scatter(
        x=pareto_optimal_points_after['params'],
        y=pareto_optimal_points_after['accuracy'],
        mode='markers+lines',
        line=dict(color='red'),
        name='Pareto Optimal',
        text=pareto_optimal_points_after.apply(lambda row: f"UID: {row['uid']}<br>FLOPs: {row['flops']}", axis=1),  # Add UID and FLOPs to hover data
        hovertemplate='%{text}<br>Params: %{x}<br>Accuracy: %{y}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title='Scatter Plot of Params vs Accuracy',
        xaxis_title='Params',
        yaxis_title='Accuracy',
        showlegend=True
    )
    
    return fig



def plot_rewards(df):
    sorted_df = df.sort_values(by='score')
    x_values = list(range(len(sorted_df)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values,
        y=sorted_df['score'],
        mode='markers',
        text=sorted_df['uid'],  # UID as hover text
        marker=dict(
            size=10,  # Adjust size for better visibility
            color=sorted_df['score'],  # Optional: use reward as color
            colorbar=dict(title='score'),
            colorscale='Viridis'
        ),
        hoverinfo='text+y',  # Show UID and Reward on hover
    ))
    
    fig.update_layout(
        title='Rewards vs. Sorted Scores',
        xaxis_title='Sorted Score Index',
        yaxis_title='Reward',
        xaxis=dict(showgrid=False),  
        yaxis=dict(showgrid=True),   
        plot_bgcolor='white',        
    )
    
    return fig

# def plot_pareto_after(df, pareto_optimal_points_after):
#     # Determine color based on Pareto optimality directly in the plot data preparation
#     colors = [1 if uid in pareto_optimal_points_after['uid'].values else 0 for uid in df['uid']]  # 1 for red, 0 for gray

#     fig = go.Figure(data=
#         go.Parcoords(
#             line=dict(
#                 color=colors,  # Applying colors to lines
#                 colorscale=[[0, 'gray'], [1, 'red']],  # Mapping 0 to gray and 1 to red
#                 showscale=False  # Optionally hide the color scale legend
#             ),
#             dimensions=[
#                 dict(label='UID', values=df['uid']),
#                 dict(label='Params', values=df['params'], tickformat=".0f"),
#                 dict(label='FLOPs', values=df['flops'], tickformat=".0f"),
#                 dict(label='Accuracy', values=df['accuracy'], tickformat=".2f")
#             ]
#         )
#     )

#     # Update layout
#     fig.update_layout(
#         title='Parallel Coordinates: UID, Params, FLOPs, and Accuracy',
#         plot_bgcolor='white'
#     )

#     return fig

def find_pareto(df, vali_config):
    pareto_optimal_uids = []
    for i, row_i in df.iterrows():
        if row_i['accuracy'] < vali_config.min_accuracy:
            continue
        if row_i['flops'] >= vali_config.max_flops:
            continue    
        is_pareto = True
        for j, row_j in df.iterrows():
            if i != j:  # Don't compare the point with itself
                if ((row_j['accuracy'] > row_i['accuracy'] and row_j['params'] <= row_i['params'] and row_j['flops'] <= row_i['flops']) or
                    (row_j['accuracy'] >= row_i['accuracy'] and row_j['params'] < row_i['params'] and row_j['flops'] <= row_i['flops']) or
                    (row_j['accuracy'] >= row_i['accuracy'] and row_j['params'] <= row_i['params'] and row_j['flops'] < row_i['flops'])):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_optimal_uids.append(row_i['uid'])
    return pareto_optimal_uids


def should_skip_evaluation(df, uid):
    if df.loc[df['uid'] == uid, 'evaluate'].values[0]:
            return True
    return False

def append_row(df, row_data):
    # Check if the uid exists in the DataFrame
    existing_row_index = df.index[df['uid'] == row_data['uid']].tolist()

    if existing_row_index:
        # Check if the commit value is different
        index = existing_row_index[0]
        if df.loc[index, 'commit'] != row_data['commit']:
            # Update the existing row
            df.loc[index] = row_data
    else:
        # If uid does not exist, append the new row
        new_row = pd.DataFrame([row_data])
        df = pd.concat([df, new_row], ignore_index=True)

    return df


def update_row(df, uid, params=None, accuracy=None, evaluate=None, pareto=None, flops =None, block=None, ext_idx= None, miner_lr = None):
    # Check if the uid exists in the DataFrame
    existing_row_index = df.index[df['uid'] == uid].tolist()

    if existing_row_index:
        # If the uid exists, update the specified fields
        index = existing_row_index[0]
        if params is not None:
            df.at[index, 'params'] = params
        if flops is not None:
            df.at[index, 'flops'] = flops
        if accuracy is not None:
            df.at[index, 'accuracy'] = accuracy
        if evaluate is not None:
            df.at[index, 'evaluate'] = evaluate
        if pareto is not None:
            df.at[index, 'pareto'] = pareto
        if block is not None:
            df.at[index, 'block'] = block
        if ext_idx is not None:
            df.at[index, 'ext_idx'] = ext_idx
        if miner_lr is not None:
            df.at[index, 'lr'] = miner_lr
    else:
        raise ValueError(f"UID {uid} does not exist in the DataFrame")

    return df


def filter_pareto_by_commit_date(df):
    df['commit_date'] = pd.to_datetime(df['commit_date'])
    pareto_df = df[df['pareto']]
    # Group by 'accuracy', 'params', and 'flops' and filter within groups
    for (accuracy, params, flops), group in pareto_df.groupby(['accuracy', 'params', 'flops']):
        if len(group) > 1:
            # Find the row with the oldest commit_date
            oldest_row_index = group['commit_date'].idxmin()
            # Set 'pareto' to False for all other rows with the same accuracy, params, and flops
            df.loc[(df['accuracy'] == accuracy) & (df['params'] == params) & (df['flops'] == flops) & (df.index != oldest_row_index), 'pareto'] = False
    
    return df

def filter_pareto_by_lowest_block(df):
    pareto_df = df[df['pareto']]
    # Group by 'accuracy', 'params', and 'flops' and filter within groups
    for (accuracy, params, flops), group in pareto_df.groupby(['accuracy', 'params', 'flops']):
        bt.logging.info(f"Processing group with accuracy: {accuracy}, params: {params}, flops: {flops}")
        
        if len(group) > 1:
            # First, find the row(s) with the lowest block number
            min_block = group['block'].min()
            min_block_group = group[group['block'] == min_block]
            bt.logging.info(f"Group has multiple rows. Minimum block number: {min_block}, number of rows with this block: {len(min_block_group)}")

            if len(min_block_group) > 1:
                # Check if multiple rows have the same lowest ext_idx
                min_ext_idx = min_block_group['ext_idx'].min()
                same_ext_idx_group = min_block_group[min_block_group['ext_idx'] == min_ext_idx]

                if len(same_ext_idx_group) > 1:
                    bt.logging.info(f"Multiple rows with the same lowest block and ext_idx: {min_ext_idx}. Setting 'pareto' to False for all.")
                    # Set 'pareto' to False for all rows in this group
                    df.loc[(df['accuracy'] == accuracy) & (df['params'] == params) & (df['flops'] == flops), 'pareto'] = False
                else:
                    # If only one row has the lowest ext_idx, keep it as Pareto and mark others as non-Pareto
                    lowest_ext_idx_row_index = min_block_group['ext_idx'].idxmin()
                    bt.logging.info(f"Lowest ext_idx: {min_ext_idx} at index: {lowest_ext_idx_row_index}. Marking others as non-Pareto.")
                    df.loc[(df['accuracy'] == accuracy) & (df['params'] == params) & (df['flops'] == flops) & (df.index != lowest_ext_idx_row_index), 'pareto'] = False
            else:
                # If only one row has the lowest block, mark others as not Pareto
                bt.logging.info(f"Only one row has the minimum block number. Marking others as non-Pareto")
                df.loc[(df['accuracy'] == accuracy) & (df['params'] == params) & (df['flops'] == flops) & (df.index != min_block_group.index[0]), 'pareto'] = False
                bt.logging.info(f"Marked non-Pareto for group with accuracy: {accuracy}, params: {params}, flops: {flops} based on block number")
    
    bt.logging.info("Finished processing all groups.")
    return df


def load_model(model_dir):
    try:
        model = torch.jit.load(model_dir)
        bt.logging.info("Torch script model loaded using torch.jit.load")
        return model
    except Exception as e:
        bt.logging.warning(f"torch.jit.load failed with error: {e}")
        try:
            model = torch.load(model_dir)
            bt.logging.info("Model loaded using torch.load")
            return model
        except Exception as jit_e:
            bt.logging.error(f"torch.load also failed with error: {jit_e}")
            raise  #

def validate_pareto(df, validated_uids, trainer, vali_config: ValidationConfig):
    changes_made = True
    while changes_made:
        changes_made = False
        # Get the current Pareto optimal points excluding validated ones
        pareto_candidates = df[df['pareto'] & ~df['uid'].isin(validated_uids)].copy()  
        bt.logging.info(f"pareto_candidates_before: {pareto_candidates}")
        
        if pareto_candidates.empty:
            break  # Exit loop if no candidates left to validate

        for index, row in pareto_candidates.iterrows():
            uid = row['uid']
            if df.loc[df['uid'] == uid, 'vali_evaluated'].values[0]:
                bt.logging.info(f"UID: {uid} has already been vali_evaluated.")
                continue
            else:
                bt.logging.info(f"UID: {uid} is being evaluated.")
            
            original_accuracy = row['accuracy']
            model_dir = df.loc[df['uid'] == uid, 'local_model_dir'].values[0]
            try:
                model = load_model(model_dir)
                # trainer.__init__(epochs=vali_config.train_epochs)
                train_lr = df[df['uid'] == uid]['lr'].iloc[0]
                trainer = ValiTrainer(epochs=vali_config.train_epochs, learning_rate=train_lr)
                trainer.initialize_weights(model)
                retrained_model = trainer.train(model)
                new_accuracy = math.floor(trainer.test(retrained_model))
                bt.logging.info(f"acc_after_retrain: {new_accuracy}")
                if new_accuracy >= original_accuracy:
                    df.loc[df['uid'] == uid, 'accuracy'] = new_accuracy
                    df.loc[df['uid'] == uid, 'pareto'] = True  # Reward the model if it passes the check
                else:
                    df.loc[df['uid'] == uid, 'accuracy'] = new_accuracy  # Update the accuracy
                    df.loc[df['uid'] == uid, 'pareto'] = False  # Mark as not Pareto optimal anymore

                validated_uids.add(uid)  # Add to validated to prevent revalidation
                df.loc[df['uid'] == uid, 'vali_evaluated'] = True  # Set the vali_evaluated flag to True for the processed model

                # Recalculate the Pareto optimal points
                new_pareto_optimal_uids = find_pareto(df, vali_config)
                bt.logging.info(f"pareto_candidates_after: {new_pareto_optimal_uids}")
                df['pareto'] = False  # Reset all Pareto flags
                df.loc[df['uid'].isin(new_pareto_optimal_uids), 'pareto'] = True  # Set new Pareto optimal points
                
                if new_pareto_optimal_uids:
                    changes_made = True  # Set changes_made to re-evaluate new Pareto front
            except Exception as e:
                bt.logging.error(f"validate_pareto error: {e}")
                # cant validate the model and remove accuracy
                df.loc[df['uid'] == uid, 'accuracy'] = 0
                df.loc[df['uid'] == uid, 'vali_evaluated'] = False
                df.loc[df['uid'] == uid, 'pareto'] = False

                # bt.logging.error(traceback.format_exc())

    # First filter those have same params and acc by commit date then set rewards 
    df = filter_pareto_by_lowest_block(df)
    final_pareto_indices = df[df['pareto']].index
    df.loc[final_pareto_indices, 'reward'] = True

    return df


def assign_rewards_to_eval_frame(df, rewarded_uids, rewards):
    # Initialize or reset all rewards to 0.0
    df['score'] = 0.0

    # Update the rewards for rewarded UIDs
    for uid, reward in zip(rewarded_uids, rewards):
        df.loc[df['uid'] == uid, 'score'] = reward

    return df



def calculate_exponential_rewards(df):
    rewarded_models = df[df['reward'] == True][['uid', 'accuracy', 'params', 'flops']]

    rewarded_models['norm_accuracy'] = (rewarded_models['accuracy'] - rewarded_models['accuracy'].min()) / (rewarded_models['accuracy'].max() - rewarded_models['accuracy'].min())
    rewarded_models['norm_params'] = (rewarded_models['params'] - rewarded_models['params'].min()) / (rewarded_models['params'].max() - rewarded_models['params'].min())
    rewarded_models['norm_flops'] = (rewarded_models['flops'] - rewarded_models['flops'].min()) / (rewarded_models['flops'].max() - rewarded_models['flops'].min())

    # By using negative coefficients for these values (-0.2), the formula decreases the combined score for models with high params and flops
    rewarded_models['combined_score'] = rewarded_models['norm_accuracy'] - 0.25 * rewarded_models['norm_params'] - 0.5 * rewarded_models['norm_flops']

    # Apply np.exp to the combined score for scaling
    exp_scores = np.exp(rewarded_models['combined_score'])

    # Normalize the scaled values so they sum to 1
    total_exp_score = exp_scores.sum()
    rewarded_models['reward'] = exp_scores / total_exp_score

    rewarded_uids = rewarded_models['uid'].tolist()
    rewards = rewarded_models['reward'].tolist()

    return rewarded_uids, rewards



def filter_columns(df):
    # df['commit_date'] = df['commit_date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if isinstance(x, pd.Timestamp) else x)
    
    columns = ['uid', 'params','flops', 'accuracy', 'pareto', 'reward', 'hf_account','commit','eval_date','score','block']
    # Create a new DataFrame with only the specified columns and reset the index
    new_df = df[columns].reset_index(drop=True)
    return new_df


# async def async_wandb_update(fig, hotkey, valiconfig, wandb_df):
#     await asyncio.to_thread(wandb_update, fig, hotkey, valiconfig, wandb_df)

def wandb_update(plot, reward_plot, hotkey, valiconfig:ValidationConfig, wandb_df):
    # Log the Plotly figure to wandb
    # wandb.log({"plotly_plot": wandb.Plotly(plot)})
    wandb.log({"reward_plot": wandb.Plotly(reward_plot)})
    # Convert commit_date to string format only if it's a datetime object
    # wandb_df['commit_date'] = wandb_df['commit_date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if isinstance(x, pd.Timestamp) else x)
    # Log the DataFrame to wandb
    wandb.log({"dataframe": wandb.Table(dataframe=wandb_df)})

# Function to check column changes
def has_columns_changed(df1, df2):
    columns_to_check = ['params', 'accuracy', 'pareto', 'flops']
    for column in columns_to_check:
        if not df1[column].equals(df2[column]):
            return True
    return False

def calc_flops(model):
    dummy_input = torch.randn(1, 3, 32, 32).cuda()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True,with_flops=True) as prof:
        with record_function("model_inference"):
            model(dummy_input)
    total_mflops = round(prof.key_averages().total_average().flops / 1e6)
    return total_mflops

def round_to_nearest_significant(x, n=2):
    if x == 0:
        return 0
    else:
        magnitude = int(math.floor(math.log10(abs(x))))
        factor = 10 ** magnitude
        return round(x / factor, n-1) * factor

def round_flops_to_nearest_significant(flops):
    if flops == 0:
        return 0
    else:
        # Determine the number of digits in the integer part of the FLOPs value
        num_digits = len(str(int(abs(flops))))

        # Decide whether to round to 1 or 2 significant digits
        if num_digits <= 7:
            n = 1
        else:
            n = 2

        # Calculate the magnitude and factor for rounding
        magnitude = int(math.floor(math.log10(abs(flops))))
        factor = 10 ** magnitude

        return round(flops / factor, n-1) * factor


def calc_flops_onnx(model):
    fixed_dummy_input = torch.randn(1, 3, 32, 32).cuda()
    onnx_path = "cache/tmp.onnx"
    profile_path = "cache/profile.txt"
    torch.onnx.export(model,
                  fixed_dummy_input,
                  onnx_path,
                  export_params=True,
                  opset_version=12,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes=None)  # No dynamic axes for profiling
    onnx_tool.model_profile(onnx_path,save_profile=profile_path)
    with open(profile_path, 'r') as file:
        profile_content = file.read()

    # Regular expression to find the total Forward MACs
    match = re.search(r'Total\s+_\s+([\d,]+)\s+100%', profile_content)

    if match:
        total_macs = match.group(1)
        total_macs = int(total_macs.replace(',', ''))
        total_macs = round_flops_to_nearest_significant(total_macs)
    return total_macs



def get_wandb_api_key():
    return os.getenv('WANDB_API_KEY') 


def get_index_in_extrinsics(block_data, hotkey):
    if not block_data:
        return None

    # Check each extrinsic in the block for the set_commitment by the provided hotkey.
    # Hotkeys can only set_commitment once every 20 minutes so just take the first we see.
    for idx, extrinsic in enumerate(block_data["extrinsics"]):
        # Check function name first, otherwise it may not have an address.
        if (
            extrinsic["call"]["call_function"]["name"] == "set_commitment"
            and extrinsic["address"] == hotkey
        ):
            return idx

    # This should never happen since we already confirmed there was metadata for this block.
    bt.logging.trace(
        f"Did not find any set_commitment for block {block} by hotkey {hotkey}"
    )
    return None


async def fetch_block_data(self, model_metadata):
    block_number = int(model_metadata.block)
    bt.logging.info(f"Attempting to fetch block: {block_number}")

    latest_block = self.archive.substrate.get_chain_head()
    bt.logging.info(f"Latest block on the node: {latest_block}")

    attempts = 3
    block_data = None

    for attempt in range(attempts):
        bt.logging.info(f"Fetch attempt {attempt + 1} for block: {block_number}")

        # Try fetching the block (synchronously)
        block_data = self.archive.substrate.get_block(block_number=block_number)

        if block_data:
            break  # Exit loop if block data is successfully retrieved

        bt.logging.info(f"Block {block_number} not found. Trying to fetch block hash...")
        block_hash = self.archive.substrate.get_block_hash(block_number)
        bt.logging.info(f"Block hash for block {block_number}: {block_hash}")

        if block_hash:
            block_data = self.archive.substrate.get_block(block_hash=block_hash)
            bt.logging.info(f"Block data retrieved using block hash: {block_data}")

            if block_data:
                break  # Exit loop if block data is successfully retrieved
        else:
            bt.logging.info("Block hash could not be retrieved.")

    if not block_data:
        raise ValueError(f"No block is available in chain for block: {block_number}")

    return block_data



async def get_metadata(metadata_store, hotkey):
    """Get metadata about a model by hotkey"""
    return await metadata_store.retrieve_model_metadata(hotkey)

async def forward(self):
    """
    The forward function is called by the validator every time step.
    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.
    """


    wandb_api_key = get_wandb_api_key()
    if wandb_api_key is None:
        bt.logging.error("Environment variable WANDB_API_KEY not found. Please set it before running the script.")
        return
    vali_config = ValidationConfig()
    trainer = ValiTrainer(epochs=vali_config.train_epochs)
    metadata_store = ChainModelMetadataStore(self.subtensor, self.wallet, self.config.netuid)
    hg_model_store = HuggingFaceModelStore()
    # Initialize wandb run with resume allowed, using the hotkey as the run ID
    try:
        wandb.init(project=vali_config.wandb_project, entity=vali_config.wandb_entitiy, resume='allow', id=str(self.wallet.hotkey.ss58_address))
    except Exception as e:
        bt.logging.error(f"Wandb init error: {e}")    
    copy_eval_frame = self.eval_frame.copy()
    miner_lr = vali_config.learning_rate
    for uid in range(self.metagraph.n.item()):
        hotkey = self.metagraph.hotkeys[uid]
        bt.logging.info(f"Reading uid: {uid} {hotkey} ---------")
        try:
            model_metadata =  await metadata_store.retrieve_model_metadata(hotkey)
            if model_metadata is None:
                raise ValueError(f"No metadata is avaiable in chain for miner:{uid}")
            bt.logging.info(f"Model Metadatdata: Hash: {model_metadata.id.hash}, commit:{model_metadata.id.commit}, learning rate: {model_metadata.id.learning_rate}")
            if model_metadata.id.learning_rate is None:
                miner_lr = vali_config.learning_rate
                bt.logging.info(f"No learning rate was submitted. Using default learning rate.")
            else:
                bt.logging.info(f"Stting learning rate to: {model_metadata.id.learning_rate}")
                miner_lr = float(model_metadata.id.learning_rate)

            # block_data = await fetch_block_data(self, model_metadata)
            # ext_idx = get_index_in_extrinsics(block_data,hotkey)
            ext_idx = np.iinfo(np.int32).max
            # bt.logging.info(f"ext_idx: {ext_idx}")

            model_with_hash, commit_date = await hg_model_store.download_model(model_metadata.id, local_path='cache', model_size_limit= vali_config.max_download_file_size)
            # bt.logging.info(f"hash_in_metadata: {model_metadata.id.hash}, {model_with_hash.id.hash}, {model_with_hash.pt_model},{model_with_hash.id.commit}")
            bt.logging.info(f"HF account: {model_metadata.id.namespace}/{model_metadata.id.name}, commitdate:{commit_date}'block:'{model_metadata.block}")
            if model_metadata.id.hash != model_with_hash.id.hash:
                # raise ValueError(f"Hash mismatch: metadata hash {model_metadata.id.hash} != downloaded model hash {model_with_hash.id.hash}")
                raise ValueError(f"Hash mismatch: metadata hash {model_metadata.id.hash[-8:]} != downloaded model hash {model_with_hash.id.hash[-8:]}")

            new_row = {
                'uid': uid,
                'local_model_dir': model_with_hash.pt_model,
                'commit_date': commit_date,
                'commit': model_with_hash.id.commit,
                'eval_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'params': np.iinfo(np.int32).max,  # Use np.iinfo() for integer types
                'flops': np.iinfo(np.int32).max,   # Use np.iinfo() for integer types
                'block': pd.NA,
                'accuracy': 0.0,
                'evaluate': False,
                'pareto': False,
                'reward': False,
                'vali_evaluated': False,
                'hf_account': model_metadata.id.namespace + "/" + model_metadata.id.name,
                'score': 0.0,
                'ext_idx': np.iinfo(np.int32).max,
                'lr': miner_lr
            }
            self.eval_frame = append_row(self.eval_frame, new_row)
            # print(self.eval_frame)
            if should_skip_evaluation(self.eval_frame, uid):
                bt.logging.info(f"Already evaluated the model for uid: {uid}")
                existing_accuracy = self.eval_frame.loc[self.eval_frame['uid'] == uid, 'accuracy'].values[0]
                rounded_accuracy = int(math.floor(existing_accuracy))
                model = load_model(model_with_hash.pt_model)
                validate_model_tensors(model)
                params = sum(param.numel() for param in model.parameters())
                params = round_to_nearest_significant(params,1)
                # flops = calc_flops(model)
                macs = calc_flops_onnx(model)
                self.eval_frame = update_row(self.eval_frame, uid,flops=macs, params = params,accuracy=rounded_accuracy, block = int(model_metadata.block), ext_idx= ext_idx, miner_lr = miner_lr)
                bt.logging.info(f"Params: {params} RoundedACC: {rounded_accuracy} MACS: {macs}")
                continue

            # print(self.eval_frame)
            # model = torch.load(model_with_hash.pt_model)
            model = load_model(model_with_hash.pt_model)
            validate_model_tensors(model)
            acc = math.floor(trainer.test(model))
            # analysis = ModelAnalysis(model) ToDo: This has issue with torch script
            params = sum(param.numel() for param in model.parameters())
            params = round_to_nearest_significant(params,1)
            # flops = calc_flops(model)
            macs = calc_flops_onnx(model)
            self.eval_frame = update_row(self.eval_frame, uid,flops=macs, accuracy = acc,params = params, evaluate = True,block = int(model_metadata.block), ext_idx= ext_idx, miner_lr = miner_lr)
            bt.logging.info(f"Params: {params} MACS: {macs}") 
            torch.cuda.empty_cache()



            

            # trainer.initialize_weights(model)
            # acc = trainer.test(model)
            # bt.logging.info(f"acc_after_rest: {acc}")
            # retrained_model = trainer.train(model)
            # acc = trainer.test(retrained_model)
            # bt.logging.info(f"acc_after_retrain: {acc}")
            # self.eval_frame = update_row(self.eval_frame, uid, accuracy = acc)
            # self.save_validator_state()
        except ReadTimeout as e:
            bt.logging.error(f"ReadTimeout on uid {uid}: {e}")
            bt.logging.error(traceback.format_exc())
        except Exception as e:
            error_message = str(e)
            if "EOF occurred in violation of protocol" in error_message:
                bt.logging.error(f"Evaluation Error on uid {uid} : {e}")
                bt.logging.error(traceback.format_exc())
            elif "ReadTimeout" in error_message:
                bt.logging.error(f"Evaluation Error on uid {uid} : {e}")
                bt.logging.error(traceback.format_exc())
            else:
                bt.logging.error(f"Evaluation Error on uid {uid} : {e}")
                # bt.logging.error(traceback.format_exc())
                if uid in self.eval_frame['uid'].values:
                    bt.logging.warning(f"Removing UID: {uid}")
                    self.eval_frame = self.eval_frame[self.eval_frame['uid'] != uid]


    try:       
        # Calculate Pareto optimal indices
        # params = self.eval_frame['params'].tolist()
        # accuracy = self.eval_frame['accuracy'].tolist()
        pareto_optimal_uids = find_pareto(self.eval_frame, vali_config)
        # reset
        self.eval_frame['pareto'] = False 
        self.eval_frame['reward'] = False 
        # Set Pareto flag to True for Pareto optimal points
        self.eval_frame.loc[self.eval_frame['uid'].isin(pareto_optimal_uids), 'pareto'] = True
        # Print Pareto optimal points before validation
        pareto_optimal_points_before = self.eval_frame[self.eval_frame['pareto']]
        pareto_tuples_before = list(pareto_optimal_points_before[['uid', 'params', 'accuracy']].itertuples(index=False, name=None))
        bt.logging.info(f"Pareto optimal points before validation:{pareto_tuples_before}")
        # Validate and adjust Pareto optimal points
        validated_uids = set()
        self.eval_frame = validate_pareto(self.eval_frame, validated_uids, trainer, vali_config)


    
        rewarded_uids, rewards = calculate_exponential_rewards(self.eval_frame ) 
        self.eval_frame = assign_rewards_to_eval_frame(self.eval_frame, rewarded_uids, rewards)
        # rewarded_uids = self.eval_frame[self.eval_frame['reward'] == True]['uid'].tolist()
        # num_rewarded = len(rewarded_uids)
        # if num_rewarded > 0:
        #     rewards = [1.0 / num_rewarded] * num_rewarded
        # else:
        #     rewards = []
        bt.logging.info(f"Rewarded_uids: {rewarded_uids}")
        bt.logging.info(f"Rewards: {rewards}")
        self.update_scores(torch.FloatTensor(rewards).to(self.device), rewarded_uids)
        self.save_validator_state()
        pareto_optimal_points_after = self.eval_frame[self.eval_frame['pareto']]
        bt.logging.info("**********************************")
        print(self.eval_frame)
        # print(copy_eval_frame)
        bt.logging.info("**********************************")
        if has_columns_changed(self.eval_frame, copy_eval_frame):
            fig = plot_pareto_after(self.eval_frame , pareto_optimal_points_after)
            fig_reward = plot_rewards(self.eval_frame)
            wandb_df = filter_columns(self.eval_frame)
            # bt.logging.info(wandb_df)
            # wandb_task = asyncio.create_task(async_wandb_update(fig, self.wallet.hotkey.ss58_address, vali_config, wandb_df))
            # await asyncio.wait([wandb_task], timeout=30)
            wandb_update(fig,fig_reward,self.wallet.hotkey.ss58_address,vali_config,wandb_df)
            # fig.show()

        wandb.finish()

        # torch.FloatTensor(rewards).to(self.device), uids, msgs

    except Exception as e:
        bt.logging.error(f"Unexpected error: {e}")
        bt.logging.error(traceback.format_exc())
    