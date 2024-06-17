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

import pandas as pd
import requests
import bittensor as bt  # Assuming this is the correct way to import bittensor in your context
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

def plot_pareto_after(df, pareto_optimal_points_after):
    fig = go.Figure()

    # Plot all points
    fig.add_trace(go.Scatter(
        x=df['params'],
        y=df['accuracy'],
        mode='markers',
        name='All Points',
        text=df['uid'],  # Add UID to hover data
        hovertemplate='UID: %{text}<br>Params: %{x}<br>Accuracy: %{y}<extra></extra>'
    ))

    # Sort and plot Pareto optimal points after validation
    pareto_optimal_points_after = pareto_optimal_points_after.sort_values(by='params')
    fig.add_trace(go.Scatter(
        x=pareto_optimal_points_after['params'],
        y=pareto_optimal_points_after['accuracy'],
        mode='markers+lines',
        line=dict(color='red'),
        name='Pareto Optimal',
        text=pareto_optimal_points_after['uid'],  # Add UID to hover data
        hovertemplate='UID: %{text}<br>Params: %{x}<br>Accuracy: %{y}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title='Scatter Plot of Params vs Accuracy',
        xaxis_title='Params',
        yaxis_title='Accuracy',
        showlegend=True
    )
    
    return fig


# Function to find Pareto optimal points
def find_pareto(accuracy, parameters):
    pareto_optimal_indices = []
    for i in range(len(accuracy)):
        is_pareto = True
        for j in range(len(accuracy)):
            if (accuracy[j] > accuracy[i] and parameters[j] <= parameters[i]) or (accuracy[j] >= accuracy[i] and parameters[j] < parameters[i]):
                is_pareto = False
                break
        if is_pareto:
            pareto_optimal_indices.append(i)
    return pareto_optimal_indices


def should_skip_evaluation(df, uid):
    # matching_rows = df.loc[df['uid'] == uid, 'evaluate']
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


def update_row(df, uid, params=None, accuracy=None, evaluate=None, pareto=None):
    # Check if the uid exists in the DataFrame
    existing_row_index = df.index[df['uid'] == uid].tolist()

    if existing_row_index:
        # If the uid exists, update the specified fields
        index = existing_row_index[0]
        if params is not None:
            df.at[index, 'params'] = params
        if accuracy is not None:
            df.at[index, 'accuracy'] = accuracy
        if evaluate is not None:
            df.at[index, 'evaluate'] = evaluate
        if pareto is not None:
            df.at[index, 'pareto'] = pareto
    else:
        raise ValueError(f"UID {uid} does not exist in the DataFrame")

    return df


def validate_pareto(df, validated_uids, trainer):
    changes_made = True
    while changes_made:
        changes_made = False
        pareto_candidates = df[df['pareto'] & ~df['uid'].isin(validated_uids)].copy()  # Get the current Pareto optimal points excluding validated ones
        
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
                model = torch.load(model_dir)
                trainer.initialize_weights(model)
                retrained_model = trainer.train(model)
                new_accuracy = trainer.test(retrained_model)
                bt.logging.info(f"acc_after_retrain: {new_accuracy}")
                if new_accuracy >= original_accuracy:
                    df.loc[df['uid'] == uid, 'reward'] = True  # Reward the model if it passes the check
                else:
                    df.loc[df['uid'] == uid, 'accuracy'] = new_accuracy  # Update the accuracy
                    df.loc[df['uid'] == uid, 'pareto'] = False  # Mark as not Pareto optimal anymore

                validated_uids.add(uid)  # Add to validated to prevent revalidation
                df.loc[df['uid'] == uid, 'vali_evaluated'] = True  # Set the vali_evaluated flag to True for the processed model

                # Recalculate the Pareto optimal points
                new_pareto_optimal_indices = find_pareto(df['accuracy'].tolist(), df['params'].tolist())
                df['pareto'] = False  # Reset all Pareto flags
                df.loc[new_pareto_optimal_indices, 'pareto'] = True  # Set new Pareto optimal points
                
                if new_pareto_optimal_indices:
                    changes_made = True  # Set changes_made to re-evaluate new Pareto front
            except Exception as e:
                bt.logging.error(f"validate_pareto error: {e}")
                bt.logging.error(traceback.format_exc())

    # After evaluating, assign rewards to any models in the Pareto front
    final_pareto_indices = df[df['pareto']].index
    df.loc[final_pareto_indices, 'reward'] = True

    return df



def wandb_update(plot, hotkey, valiconfig:ValidationConfig):
    api_key = os.getenv('WANDB_API_KEY')
    if api_key is not None:
    # Log in to wandb using the API key from the environment variable
        wandb.login(key=api_key)
    else:
        print("Environment variable WANDB_API_KEY not found. Please set it before running the script.")
    run_id = hotkey
    # Initialize wandb run with resume
    wandb.init(project=valiconfig.wandb_project, entity=valiconfig.wandb_entitiy, resume='allow', id=run_id)

    # Log the Plotly figure to wandb
    wandb.log({"plotly_plot": wandb.Plotly(plot)})

    # Finish the wandb run
    wandb.finish()


# Function to check column changes
def has_columns_changed(df1, df2):
    columns_to_check = ['params', 'accuracy', 'pareto']
    for column in columns_to_check:
        if not df1[column].equals(df2[column]):
            return True
    return False
  


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

    vali_config = ValidationConfig()
    trainer = ValiTrainer(epochs=vali_config.train_epochs)
    metadata_store = ChainModelMetadataStore(self.subtensor, self.wallet, self.config.netuid)
    hg_model_store = HuggingFaceModelStore()
    copy_eval_frame = self.eval_frame.copy()
    for uid in range(self.metagraph.n.item()):
        bt.logging.error(f"--------------------")
        hotkey = self.metagraph.hotkeys[uid]
        bt.logging.info(f"uid {uid} {hotkey}")
        try:
            model_metadata =  await metadata_store.retrieve_model_metadata(hotkey)
            model_with_hash = await hg_model_store.download_model(model_metadata.id, local_path='cache', model_size_limit= vali_config.max_download_file_size)
            bt.logging.info(f"hash_in_metadata: {model_metadata.id.hash}, {model_with_hash.id.hash}, {model_with_hash.pt_model},{model_with_hash.id.commit}")
            
            if model_metadata.id.hash != model_with_hash.id.hash:
                raise ValueError(f"Hash mismatch: metadata hash {model_metadata.id.hash} != downloaded model hash {model_with_hash.id.hash}")

            new_row = {
                'uid': uid,
                'local_model_dir': model_with_hash.pt_model,
                'commit': model_with_hash.id.commit,
                'params': float('inf'),
                'accuracy': 0.0,
                'evaluate': False,
                'pareto': False,
                'reward': False,
                'vali_evaluated':False,
                
            }
            self.eval_frame = append_row(self.eval_frame, new_row)
            print(self.eval_frame)
            if should_skip_evaluation(self.eval_frame, uid):
                bt.logging.info(f"already evaluated the model")
                continue

            # print(self.eval_frame)
            model = torch.load(model_with_hash.pt_model)
            acc = trainer.test(model)
            analysis = ModelAnalysis(model)
            params, macs, flops = analysis.get_analysis()
            self.eval_frame = update_row(self.eval_frame, uid, accuracy = acc,params = params, evaluate = True)
            bt.logging.info(f"acc_before: {acc}") 
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
            bt.logging.error(f"Unexpected error: {e}")
            bt.logging.error(traceback.format_exc())
            # traceback.print_exc()
    try:       
        # Calculate Pareto optimal indices
        params = self.eval_frame['params'].tolist()
        accuracy = self.eval_frame['accuracy'].tolist()
        pareto_optimal_indices = find_pareto(accuracy, params)

        # Set Pareto flag to True for Pareto optimal points
        self.eval_frame.loc[pareto_optimal_indices, 'pareto'] = True
        # Print Pareto optimal points before validation
        pareto_optimal_points_before = self.eval_frame[self.eval_frame['pareto']]
        pareto_tuples_before = list(pareto_optimal_points_before[['uid', 'params', 'accuracy']].itertuples(index=False, name=None))
        print("Pareto optimal points before validation:", pareto_tuples_before)
        # Validate and adjust Pareto optimal points
        validated_uids = set()
        self.eval_frame = validate_pareto(self.eval_frame, validated_uids, trainer)


    
        
        rewarded_uids = self.eval_frame[self.eval_frame['reward'] == True]['uid'].tolist()
        num_rewarded = len(rewarded_uids)
        if num_rewarded > 0:
            rewards = [1.0 / num_rewarded] * num_rewarded
        else:
            rewards = []
        bt.logging.info(f"Rewarded_uids: {rewarded_uids}")
        bt.logging.info(f"rewards: {rewards}")
        self.update_scores(torch.FloatTensor(rewards).to(self.device), rewarded_uids)
        self.save_validator_state()
        pareto_optimal_points_after = self.eval_frame[self.eval_frame['pareto']]
        print("**********************************")
        print(self.eval_frame)
        print(copy_eval_frame)
        print("**********************************")
        if has_columns_changed(self.eval_frame, copy_eval_frame):
            fig = plot_pareto_after(self.eval_frame , pareto_optimal_points_after)
            wandb_update(fig,"firsttest",vali_config)
            fig.show()



        # torch.FloatTensor(rewards).to(self.device), uids, msgs

    except Exception as e:
        bt.logging.error(f"Unexpected error: {e}")




    
    # Send a GET request to the server
    # response = requests.get(url)


    # TODO(developer): Define how the validator selects a miner to query, how often, etc.
    # get_random_uids is an example method, but you can replace it with your own.
    # miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    # # The dendrite client queries the network.
    # responses = await self.dendrite(
    #     # Send the query to selected miner axons in the network.
    #     axons=[self.metagraph.axons[uid] for uid in miner_uids],
    #     # Construct a dummy query. This simply contains a single integer.
    #     synapse=Dummy(dummy_input=self.step),
    #     # All responses have the deserialize function called on them before returning.
    #     # You are encouraged to define your own deserialization function.
    #     deserialize=True,
    # )

    # # Log the results for monitoring purposes.
    # bt.logging.info(f"Received responses: {responses}")

    # # TODO(developer): Define how the validator scores responses.
    # # Adjust the scores based on responses from miners.
    # rewards = get_rewards(self, query=self.step, responses=responses)

    # bt.logging.info(f"Scored responses: {rewards}")
    # # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    # self.update_scores(rewards, miner_uids)
