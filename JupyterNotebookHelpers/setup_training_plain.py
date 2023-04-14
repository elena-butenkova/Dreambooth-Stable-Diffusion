import os
from git import Repo

from dreambooth_helpers.joepenna_dreambooth_config import JoePennaDreamboothConfigSchemaV1
from JupyterNotebookHelpers.download_model import SDModelOption

MAGIC_TRAINING_CONST = 101
FLIP_PERCENT = 0.5


def get_number_of_images(training_images_folder_path: str = "./training_images") -> int:
    image_extensions = ['.jpg', '.jpeg', '.png']
    return len([
        file_name for file_name in os.listdir(training_images_folder_path)
        if any(file_name.endswith(ext) for ext in image_extensions)
    ])


def download_regularization_imgs(dataset: str) -> str:
    repo_name = f"Stable-Diffusion-Regularization-Images-{dataset}"
    path_to_reg_images = os.path.join(repo_name, dataset)

    if not os.path.exists(path_to_reg_images):
        print(f"Downloading regularization images for {dataset}. Please wait...")
        Repo.clone_from(f"https://github.com/djbielejeski/{repo_name}.git", repo_name)

        print(f"✅ Regularization images for {dataset} downloaded successfully.")
    else:
        print(f"✅ Regularization images for {dataset} already exist. Skipping download...")

    return path_to_reg_images


def run(project_name: str, token: str, class_word: str, learning_rate: float, selected_model: SDModelOption,
        save_every_x_steps: int = 0, dataset: str = "artstyle",
        training_images_path: str = "./training_images", config_save_path: str = "./joepenna-dreambooth-configs"):

    number_of_pictures = get_number_of_images(training_images_path)
    regularization_images_path = download_regularization_imgs(dataset)

    config = JoePennaDreamboothConfigSchemaV1()
    config.saturate(
        project_name=project_name,
        max_training_steps=number_of_pictures * MAGIC_TRAINING_CONST,
        save_every_x_steps=save_every_x_steps,
        training_images_folder_path=training_images_path,
        regularization_images_folder_path=regularization_images_path,
        token=token,
        token_only=False,
        class_word=class_word,
        flip_percent=FLIP_PERCENT,
        learning_rate=learning_rate,
        model_repo_id=selected_model.repo_id,
        model_path=selected_model.filename,
        run_seed_everything=False,
    )

    config.save_config_to_file(
        save_path=config_save_path,
        create_active_config=True
    )
