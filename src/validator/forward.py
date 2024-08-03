# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Developer: Nima Aghli   
# Copyright © 2023 Nima Aghli

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
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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


def update_row(df, uid, params=None, accuracy=None, evaluate=None, pareto=None, flops =None):
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
    else:
        raise ValueError(f"UID {uid} does not exist in the DataFrame")

    return df


def filter_pareto_by_commit_date(df):
    # Ensure commit_date is in datetime format
    df['commit_date'] = pd.to_datetime(df['commit_date'])
    pareto_df = df[df['pareto']]
    
    # Group by 'accuracy', 'params', and 'pareto' and filter within groups
    for (accuracy, params, pareto), group in pareto_df.groupby(['accuracy', 'params', 'pareto']):
        if len(group) > 1:
            # Find the row with the oldest commit_date
            oldest_row_index = group['commit_date'].idxmin()
            
            # Set 'pareto' to False for all other rows with the same accuracy, params, and pareto status
            df.loc[(df['accuracy'] == accuracy) & (df['params'] == params) & (df['pareto'] == pareto) & (df.index != oldest_row_index), 'pareto'] = False
    
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
                trainer.__init__(epochs=vali_config.train_epochs)
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
                df.loc[df['uid'] == uid, 'vali_evaluated'] = True
                df.loc[df['uid'] == uid, 'pareto'] = False

                # bt.logging.error(traceback.format_exc())

    # First filter those have same params and acc by commit date then set rewards 
    df = filter_pareto_by_commit_date(df)
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


import numpy as np
import pandas as pd

def calculate_exponential_rewards(df):
    # Filter to include only those models marked for reward
    rewarded_models = df[df['reward'] == True][['uid', 'accuracy', 'params', 'flops']]
    
    # Normalize each metric to a 0-1 range
    rewarded_models['norm_accuracy'] = (rewarded_models['accuracy'] - rewarded_models['accuracy'].min()) / (rewarded_models['accuracy'].max() - rewarded_models['accuracy'].min())
    rewarded_models['norm_params'] = (rewarded_models['params'] - rewarded_models['params'].min()) / (rewarded_models['params'].max() - rewarded_models['params'].min())
    rewarded_models['norm_flops'] = (rewarded_models['flops'] - rewarded_models['flops'].min()) / (rewarded_models['flops'].max() - rewarded_models['flops'].min())

    # By using negative coefficients for these values (-0.2), the formula decreases the combined score for models with high params and flops
    rewarded_models['combined_score'] = rewarded_models['norm_accuracy'] - 0.25 * rewarded_models['norm_params'] - 0.5 * rewarded_models['norm_flops']

    # Apply np.log1p to the combined score for scaling
    scaled_scores = np.log1p(rewarded_models['combined_score'] - rewarded_models['combined_score'].min())

    # Normalize the scaled values so they sum to 1
    total_scaled_score = scaled_scores.sum()
    rewarded_models['reward'] = scaled_scores / total_scaled_score

    # Prepare output lists
    rewarded_uids = rewarded_models['uid'].tolist()
    rewards = rewarded_models['reward'].tolist()

    return rewarded_uids, rewards


def filter_columns(df):
    # df['commit_date'] = df['commit_date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if isinstance(x, pd.Timestamp) else x)
    
    columns = ['uid', 'params','flops', 'accuracy', 'pareto', 'reward', 'hf_account','commit','eval_date','score']
    # Create a new DataFrame with only the specified columns and reset the index
    new_df = df[columns].reset_index(drop=True)
    return new_df

def wandb_update(plot, reward_plot, hotkey, valiconfig:ValidationConfig, wandb_df):
    # Log the Plotly figure to wandb
    wandb.log({"plotly_plot": wandb.Plotly(plot)})
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


def get_wandb_api_key():
    return os.getenv('WANDB_API_KEY') 


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
    for uid in range(self.metagraph.n.item()):
        hotkey = self.metagraph.hotkeys[uid]
        bt.logging.info(f"Reading uid: {uid} {hotkey} ---------")
        try:
            model_metadata =  await metadata_store.retrieve_model_metadata(hotkey)
            if model_metadata is None:
                raise ValueError(f"No metadata is avaiable in chain for miner:{uid}")
            model_with_hash, commit_date = await hg_model_store.download_model(model_metadata.id, local_path='cache', model_size_limit= vali_config.max_download_file_size)
            # bt.logging.info(f"hash_in_metadata: {model_metadata.id.hash}, {model_with_hash.id.hash}, {model_with_hash.pt_model},{model_with_hash.id.commit}")
            bt.logging.info(f"HF account: {model_metadata.id.namespace}/{model_metadata.id.name}, commitdate:{commit_date}")
            if model_metadata.id.hash != model_with_hash.id.hash:
                # raise ValueError(f"Hash mismatch: metadata hash {model_metadata.id.hash} != downloaded model hash {model_with_hash.id.hash}")
                raise ValueError(f"Hash mismatch: metadata hash {model_metadata.id.hash[-8:]} != downloaded model hash {model_with_hash.id.hash[-8:]}")

            new_row = {
                'uid': uid,
                'local_model_dir': model_with_hash.pt_model,
                'commit_date': commit_date,
                'commit': model_with_hash.id.commit,
                'eval_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'params': float('inf'),
                'flops': float('inf'),
                'accuracy': 0.0,
                'evaluate': False,
                'pareto': False,
                'reward': False,
                'vali_evaluated':False,
                'hf_account': model_metadata.id.namespace + "/" + model_metadata.id.name, 
                
            }
            self.eval_frame = append_row(self.eval_frame, new_row)
            # print(self.eval_frame)
            if should_skip_evaluation(self.eval_frame, uid):
                bt.logging.info(f"Already evaluated the model for uid: {uid}")
                existing_accuracy = self.eval_frame.loc[self.eval_frame['uid'] == uid, 'accuracy'].values[0]
                rounded_accuracy = int(math.floor(existing_accuracy))
                model = load_model(model_with_hash.pt_model)
                params = sum(param.numel() for param in model.parameters())
                params = round(params, -3)
                flops = calc_flops(model)
                self.eval_frame = update_row(self.eval_frame, uid,flops=flops, params = params,accuracy=rounded_accuracy)
                bt.logging.info(f"Params: {params} RoundedACC: {rounded_accuracy} MFlops: {flops}")
                continue

            # print(self.eval_frame)
            # model = torch.load(model_with_hash.pt_model)
            model = load_model(model_with_hash.pt_model)
            acc = math.floor(trainer.test(model))
            # analysis = ModelAnalysis(model) ToDo: This has issue with torch script
            params = sum(param.numel() for param in model.parameters())
            params = round(params, -3)
            flops = calc_flops(model)
            self.eval_frame = update_row(self.eval_frame, uid,flops=flops, accuracy = acc,params = params, evaluate = True)
            bt.logging.info(f"Eval Acc: {acc}, Eval Params: {params}") 
            torch.cuda.empty_cache()



            

            # trainer.initialize_weights(model)
            # acc = trainer.test(model)
            # bt.logging.info(f"acc_after_rest: {acc}")
            # retrained_model = trainer.train(model)
            # acc = trainer.test(retrained_model)
            # bt.logging.info(f"acc_after_retrain: {acc}")
            # self.eval_frame = update_row(self.eval_frame, uid, accuracy = acc)
            # self.save_validator_state()
        except Exception as e:
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
            wandb_update(fig,fig_reward,self.wallet.hotkey.ss58_address,vali_config,wandb_df)
            # fig.show()

        wandb.finish()

        # torch.FloatTensor(rewards).to(self.device), uids, msgs

    except Exception as e:
        bt.logging.error(f"Unexpected error: {e}")
        bt.logging.error(traceback.format_exc())
    