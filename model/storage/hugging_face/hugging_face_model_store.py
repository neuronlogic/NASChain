import sys
import tempfile
import os
from huggingface_hub import HfApi, HfFolder, Repository,hf_hub_download
from model.data import Model, ModelId
from model.storage.disk import utils
# from transformers import AutoModelForCausalLM

from model.storage.remote_model_store import RemoteModelStore
from model.vali_config import ValidationConfig
import constants


class HuggingFaceModelStore(RemoteModelStore):
    """Hugging Face based implementation for storing and retrieving a model."""

    @classmethod
    def assert_access_token_exists(cls) -> str:
        """Asserts that the access token exists."""
        if not os.getenv("HF_ACCESS_TOKEN"):
            raise ValueError("No Hugging Face access token found to write to the hub.")
        return os.getenv("HF_ACCESS_TOKEN")

    async def upload_model(self, model: Model) -> ModelId:
        """Uploads a trained model to Hugging Face."""
        token = HuggingFaceModelStore.assert_access_token_exists()
        vali_config = ValidationConfig()
        repo_id = model.id.namespace + "/" + model.id.name
        api = HfApi()
        # # PreTrainedModel.save_pretrained only saves locally
        # commit_info = model.pt_model.push_to_hub(
        #     repo_id=model.id.namespace + "/" + model.id.name,
        #     token=token,
        #     safe_serialization=True,
        # )


        # Check if the repository exists
        try:
            api.repo_info(repo_id=repo_id, token=token)
        except Exception as e:
            if '404' in str(e):
                # Repository does not exist, create it
                api.create_repo(repo_id=repo_id, token=token)
            else:
                raise


        # Upload the model file
        api.upload_file(
            path_or_fileobj=model.pt_model,
            path_in_repo="model.pt",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        commit_info = api.model_info(repo_id=repo_id, token=token)
        model_id_with_commit = ModelId(
            namespace=model.id.namespace,
            # accuracy=model.id.accuracy,
            name=model.id.name,
            hash=model.id.hash,
            commit=commit_info.sha,  # Get the latest commit sha
        )
        print("commit infor:--", commit_info)
        
        # TODO consider skipping the redownload if a hash is already provided.
        # To get the hash we need to redownload it at a local tmp directory after which it can be deleted.
        with tempfile.TemporaryDirectory() as temp_dir:
            model_with_hash, _ = await self.download_model(model_id_with_commit, temp_dir, vali_config.max_download_file_size)
            # Return a ModelId with both the correct commit and hash.
            return model_with_hash.id

    async def download_model(
        self, model_id: ModelId, local_path: str, model_size_limit: int = sys.maxsize
    ) -> Model:
        """Retrieves a trained model from Hugging Face."""
        if not model_id.commit:
            raise ValueError("No Hugging Face commit id found to read from the hub.")

        repo_id = model_id.namespace + "/" + model_id.name

        # Check ModelInfo for the size of model.safetensors file before downloading.
        api = HfApi()
        model_info = api.model_info(
            repo_id=repo_id, revision=model_id.commit, timeout=10, files_metadata=True
        )
        commit_date =  model_info.last_modified
        size = sum(repo_file.size for repo_file in model_info.siblings)
        if size > model_size_limit:
            raise ValueError(
                f"Hugging Face repo over maximum size limit. Size {size}. Limit {model_size_limit}."
            )

        # Transformers library can pick up a model based on the hugging face path (username/model) + rev.
        # model = AutoModelForCausalLM.from_pretrained(
        #     pretrained_model_name_or_path=repo_id,
        #     revision=model_id.commit,
        #     cache_dir=local_path,
        #     use_safetensors=True,
        # )

        # Get the directory the model was stored to.
        model_dir = utils.get_hf_download_path(local_path, model_id)

        cache_dir = os.path.join(os.getcwd(), 'cache')

        # Ensure the cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        local_model_path = hf_hub_download(repo_id=repo_id, filename='model.pt',cache_dir=cache_dir)

        

        # Realize all symlinks in that directory since Transformers library does not support avoiding symlinks.
        # utils.realize_symlinks_in_directory(model_dir)

        # Compute the hash of the downloaded model.
        model_hash = utils.get_hash_of_directory(os.path.dirname(local_model_path))
        # print("model_hash:", model_hash, local_model_path)
        model_id_with_hash = ModelId(
            namespace=model_id.namespace,
            name=model_id.name,
            commit=model_id.commit,
            hash=model_hash,
            # accuracy=model_id.accuracy
        )

        return Model(id=model_id_with_hash, pt_model=local_model_path), commit_date



class HuggingFaceMinerModelStore(RemoteModelStore):
    """Hugging Face based implementation for storing and retrieving a model."""

    @classmethod
    def assert_access_token_exists(cls) -> str:
        """Asserts that the access token exists."""
        if not os.getenv("HF_ACCESS_TOKEN"):
            raise ValueError("No Hugging Face access token found to write to the hub.")
        return os.getenv("HF_ACCESS_TOKEN")

    async def upload_model(self, model: Model) -> ModelId:
        """Uploads a trained model to Hugging Face."""
        token = HuggingFaceMinerModelStore.assert_access_token_exists()
        vali_config = ValidationConfig()
        repo_id = model.id.namespace + "/" + model.id.name
        api = HfApi()
        # # PreTrainedModel.save_pretrained only saves locally
        # commit_info = model.pt_model.push_to_hub(
        #     repo_id=model.id.namespace + "/" + model.id.name,
        #     token=token,
        #     safe_serialization=True,
        # )


        try:
            # Check if the repository exists
            repo_info = api.repo_info(repo_id=repo_id, token=token)

            # If it exists and is not private, make it private
            if not repo_info.private:
                api.update_repo_visibility(repo_id=repo_id, private=True, token=token)
                print(f"Repository {repo_id} set to private.")
        except Exception as e:
            if '404' in str(e):
                # Repository does not exist, create it as private
                api.create_repo(repo_id=repo_id, token=token, private=True)
                print(f"Repository {repo_id} created in private mode.")
            else:
                raise


        # Upload the model file
        api.upload_file(
            path_or_fileobj=model.pt_model,
            path_in_repo="model.pt",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        commit_info = api.model_info(repo_id=repo_id, token=token)
        model_id_with_commit = ModelId(
            namespace=model.id.namespace,
            learning_rate=model.id.learning_rate,
            name=model.id.name,
            hash=model.id.hash,
            commit=commit_info.sha,  # Get the latest commit sha
        )
        print("commit infor:--", commit_info)
        
        # TODO consider skipping the redownload if a hash is already provided.
        # To get the hash we need to redownload it at a local tmp directory after which it can be deleted.
        with tempfile.TemporaryDirectory() as temp_dir:
            model_with_hash, _ = await self.download_model(model_id_with_commit, temp_dir, vali_config.max_download_file_size)
            # Return a ModelId with both the correct commit and hash.
            return model_with_hash.id
    
    async def set_repo_public(self, model: Model) -> ModelId:
        """Sets the repository to public."""
        token = HuggingFaceMinerModelStore.assert_access_token_exists()
        repo_id = model.id.namespace + "/" + model.id.name
        api = HfApi()
        try:
            api.update_repo_visibility(repo_id=repo_id, private=False, token=token)
            print(f"Repository {repo_id} is now public.")
        except Exception as e:
            print(f"Failed to set the repository to public: {e}")

    async def download_model(
        self, model_id: ModelId, local_path: str, model_size_limit: int = sys.maxsize
    ) -> Model:
        """Retrieves a trained model from Hugging Face."""
        if not model_id.commit:
            raise ValueError("No Hugging Face commit id found to read from the hub.")
        token = HuggingFaceMinerModelStore.assert_access_token_exists()
        repo_id = model_id.namespace + "/" + model_id.name

        # Check ModelInfo for the size of model.safetensors file before downloading.
        api = HfApi()
        model_info = api.model_info(
            repo_id=repo_id, revision=model_id.commit, timeout=10, files_metadata=True, token=token 
        )
        commit_date =  model_info.last_modified
        size = sum(repo_file.size for repo_file in model_info.siblings)
        if size > model_size_limit:
            raise ValueError(
                f"Hugging Face repo over maximum size limit. Size {size}. Limit {model_size_limit}."
            )

        # Transformers library can pick up a model based on the hugging face path (username/model) + rev.
        # model = AutoModelForCausalLM.from_pretrained(
        #     pretrained_model_name_or_path=repo_id,
        #     revision=model_id.commit,
        #     cache_dir=local_path,
        #     use_safetensors=True,
        # )

        # Get the directory the model was stored to.
        model_dir = utils.get_hf_download_path(local_path, model_id)

        cache_dir = os.path.join(os.getcwd(), 'cache')

        # Ensure the cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        local_model_path = hf_hub_download(repo_id=repo_id, filename='model.pt',cache_dir=cache_dir, token =token)

        

        # Realize all symlinks in that directory since Transformers library does not support avoiding symlinks.
        # utils.realize_symlinks_in_directory(model_dir)

        # Compute the hash of the downloaded model.
        model_hash = utils.get_hash_of_directory(os.path.dirname(local_model_path))
        # print("model_hash:", model_hash, local_model_path)
        model_id_with_hash = ModelId(
            namespace=model_id.namespace,
            name=model_id.name,
            commit=model_id.commit,
            hash=model_hash,
            learning_rate=model_id.learning_rate,
        )

        return Model(id=model_id_with_hash, pt_model=local_model_path), commit_date
