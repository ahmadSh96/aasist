import json

notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {"id": "title"},
            "source": [
                "# AASIST + AST: Audio Spoof Detection with Audio Spectrogram Transformer (Colab with Resume Support)",
                "",
                "This notebook trains the **AASIST-AST** hybrid model, combining AASIST with an Audio Spectrogram Transformer (AST) encoder. It includes **checkpointing and resume functionality** to overcome Google Colab's session limits.",
                "",
                "## Architecture Overview",
                "- **AASIST Branch**: Raw waveform → SincConv → ResNet encoder → Spectral & Temporal GATs → Graph pooling",
                "- **AST Branch**: Raw waveform → Mel Spectrogram → Patch Embedding → Transformer Encoder → CLS token",
                "- **Fusion**: Concatenation of both branches → Final binary classifier",
                "",
                "## Expected Performance",
                "The baseline AASIST achieves **EER: 0.83%, min t-DCF: 0.0275** on ASVspoof2019 LA eval set.  ",
                "The AASIST-AST hybrid is expected to improve upon this by leveraging global spectro-temporal attention from the Transformer.",
                "",
                "---",
                "**Runtime**: Use **GPU** (T4 or A100 recommended). Go to `Runtime → Change runtime type → GPU`."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "step1"},
            "source": ["## Step 1: Check GPU"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "check_gpu"},
            "outputs": [],
            "source": [
                "import torch",
                "print(f\"PyTorch version: {torch.__version__}\")",
                "print(f\"CUDA available: {torch.cuda.is_available()}\")",
                "if torch.cuda.is_available():",
                "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")",
                "    print(f\"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "step2"},
            "source": ["## Step 2: Mount Google Drive for Checkpoints"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "mount_drive"},
            "outputs": [],
            "source": [
                "from google.colab import drive",
                "drive.mount(\"/content/drive\")",
                "",
                "# Define a path in your Google Drive to save/load checkpoints",
                "DRIVE_CHECKPOINT_DIR = \"/content/drive/MyDrive/AASIST_AST_Checkpoints\"",
                "!mkdir -p {DRIVE_CHECKPOINT_DIR}",
                "print(f\"✅ Google Drive mounted. Checkpoints will be saved/loaded from: {DRIVE_CHECKPOINT_DIR}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "step3"},
            "source": ["## Step 3: Clone Repository and Install Dependencies"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "setup"},
            "outputs": [],
            "source": [
                "import os",
                "",
                "# Change to /content/ directory to ensure correct cloning path",
                "%cd /content/",
                "",
                "# Clone the forked repository if it doesn't exist",
                "if not os.path.exists(\"aasist\"):",
                "    !git clone https://github.com/ahmadSh96/aasist.git",
                "",
                "# Change into the cloned repository directory",
                "%cd aasist",
                "",
                "# Switch to the AST integration branch",
                "!git checkout feature/ast-integration",
                "!git pull origin feature/ast-integration",
                "",
                "# Install dependencies",
                "!pip install -q torchcontrib soundfile torchaudio",
                "",
                "print(\"\n✅ Setup complete!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "step4"},
            "source": [
                "## Step 4: Download ASVspoof 2019 LA Dataset",
                "",
                "> **Note**: The dataset is ~10GB. This will take several minutes.",
                "> If you already have the dataset, skip this cell and set `database_path` in the config accordingly."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "download_data"},
            "outputs": [],
            "source": [
                "import os",
                "",
                "# Ensure LA directory is created at the root of the aasist project",
                "if not os.path.exists(\"LA\"):",
                "    print(\"Downloading ASVspoof 2019 LA dataset (~10GB)...\")",
                "    !wget -q --show-progress https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip",
                "    print(\"Extracting...\")",
                "    !unzip -q LA.zip",
                "    !rm LA.zip",
                "    print(\"✅ Dataset ready!\")",
                "else:",
                "    print(\"✅ Dataset already exists, skipping download.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "step5"},
            "source": [
                "## Step 5: Verify Model Architecture",
                "",
                "Let's inspect the AASIST-AST model to confirm it loads correctly and count parameters."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "verify_model"},
            "outputs": [],
            "source": [
                "import sys",
                "sys.path.insert(0, \"./\")",
                "",
                "import json",
                "import torch",
                "",
                "# Load config",
                "with open(\"config/AASIST_AST.conf\", \"r\") as f:",
                "    config = json.load(f)",
                "",
                "model_config = config[\"model_config\"]",
                "",
                "# Load model",
                "from models.AASIST_AST import Model",
                "model = Model(model_config)",
                "",
                "# Count parameters",
                "total_params = sum(p.numel() for p in model.parameters())",
                "trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)",
                "",
                "print(f\"Model: AASIST-AST\")",
                "print(f\"Total parameters:     {total_params:,}\")",
                "print(f\"Trainable parameters: {trainable_params:,}\")",
                "",
                "# Test forward pass",
                "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"",
                "model = model.to(device)",
                "dummy_input = torch.randn(2, 64600).to(device)  # batch=2, 4 seconds at 16kHz",
                "with torch.no_grad():",
                "    features, output = model(dummy_input)",
                "",
                "print(f\"\nForward pass test:\")",
                "print(f\"  Input shape:    {dummy_input.shape}\")",
                "print(f\"  Features shape: {features.shape}\")",
                "print(f\"  Output shape:   {output.shape}\")",
                "print(\"\n✅ Model loaded and forward pass successful!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "step6"},
            "source": ["## Step 6: Train the AASIST-AST Model (with Checkpointing)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "train"},
            "outputs": [],
            "source": [
                "import os",
                "import glob",
                "import shutil",
                "",
                "# Define output directory for this Colab session",
                "COLAB_OUTPUT_DIR = \"colab_exp_result\"",
                "os.makedirs(COLAB_OUTPUT_DIR, exist_ok=True)",
                "",
                "print(f\"DRIVE_CHECKPOINT_DIR: {DRIVE_CHECKPOINT_DIR}\")",
                "print(\"Listing contents of DRIVE_CHECKPOINT_DIR:\")",
                "!ls -R {DRIVE_CHECKPOINT_DIR}",
                "",
                "# Find latest checkpoint in Google Drive",
                "latest_checkpoint = None",
                "# The main.py saves checkpoints to {COLAB_OUTPUT_DIR}/{model_tag}/weights/checkpoint.pth",
                "# When copied to Drive, the structure is DRIVE_CHECKPOINT_DIR/{model_tag}/weights/checkpoint.pth",
                "# We need to find the actual model_tag directory within COLAB_OUTPUT_DIR and copy that specific directory to DRIVE_CHECKPOINT_DIR.",
                "# The glob pattern should now look for checkpoint.pth inside any weights folder within any model_tag folder.",
                "checkpoint_files = sorted(glob.glob(f\"{DRIVE_CHECKPOINT_DIR}/**/weights/checkpoint.pth\", recursive=True), key=os.path.getmtime)",
                "print(f\"Found checkpoint files: {checkpoint_files}\")",
                "if checkpoint_files:",
                "    latest_checkpoint = checkpoint_files[-1]",
                "    print(f\"Found latest checkpoint: {latest_checkpoint}\")",
                "",
                "# Construct training command",
                "train_command_parts = [",
                "    \"python main.py\",",
                "    \"--config config/AASIST_AST.conf\",",
                "    f\"--output_dir {COLAB_OUTPUT_DIR}\",",
                "    \"--seed 1234\"",
                "]",
                "",
                "if latest_checkpoint:",
                "    train_command_parts.append(f\"--resume_checkpoint {latest_checkpoint}\")",
                "",
                "train_command = \" \".join(train_command_parts)",
                "",
                "print(\"Starting training with command:\")",
                "print(f\"!{train_command}\")",
                "!{train_command}",
                "",
                "# After training, copy results to Google Drive",
                "print(f\"\nCopying results from {COLAB_OUTPUT_DIR} to {DRIVE_CHECKPOINT_DIR}...\")",
                "# The main.py saves checkpoints to {COLAB_OUTPUT_DIR}/{model_tag}/weights/checkpoint.pth",
                "# We need to find the actual model_tag directory within COLAB_OUTPUT_DIR and copy that specific directory to DRIVE_CHECKPOINT_DIR.",
                "model_tag_dirs = glob.glob(f\"{COLAB_OUTPUT_DIR}/*_ep*_bs*\")",
                "if model_tag_dirs:",
                "    actual_output_to_copy = model_tag_dirs[0] # Assuming only one such directory is created per run",
                "    print(f\"Copying actual output directory: {actual_output_to_copy} to {DRIVE_CHECKPOINT_DIR}/\")",
                "    # Use shutil.copytree to copy the directory and its contents",
                "    # If the destination already exists, copytree will raise an error, so we need to handle that.",
                "    # A simpler approach for Colab is to copy the contents of the model_tag directory.",
                "    !cp -r {actual_output_to_copy} {DRIVE_CHECKPOINT_DIR}/",
                "else:",
                "    print(\"No model output directory found to copy.\")",
                "print(\"✅ Training complete and results saved to Google Drive!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "step7"},
            "source": ["## Step 7: Evaluate the Best Model"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "evaluate_model"},
            "outputs": [],
            "source": [
                "import os",
                "import glob",
                "import torch",
                "import json",
                "import sys",
                "sys.path.insert(0, \"./\")",
                "from main import get_model, get_loader, produce_evaluation_file, calculate_tDCF_EER",
                "from pathlib import Path",
                "",
                "# Load config",
                "with open(\"config/AASIST_AST.conf\", \"r\") as f:",
                "    config = json.load(f)",
                "",
                "model_config = config[\"model_config\"]",
                "track = config[\"track\"]",
                "database_path = Path(config[\"database_path\"])",
                "prefix_2019 = \"ASVspoof2019.{}\".format(track)",
                "eval_trial_path = (",
                "    database_path /",
                "    \"ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt\".format(",
                "        track, prefix_2019))",
                "",
                "# Find the latest best.pth in Google Drive",
                "best_model_files = sorted(glob.glob(f\"{DRIVE_CHECKPOINT_DIR}/**/weights/best.pth\", recursive=True), key=os.path.getmtime)",
                "",
                "if best_model_files:",
                "    best_model_path = best_model_files[-1]",
                "    print(f\"Found best model: {best_model_path}\")",
                "",
                "    device = \"cuda\" if torch.cuda.is_available() else \"cpu\"",
                "    model = get_model(model_config, device)",
                "    model.load_state_dict(torch.load(best_model_path, map_location=device))",
                "    print(\"Model loaded for evaluation.\")",
                "",
                "    # Define output paths for evaluation scores",
                "    model_tag_from_path = Path(best_model_path).parent.parent.name",
                "    eval_output_dir = Path(\"colab_eval_result\") / model_tag_from_path",
                "    os.makedirs(eval_output_dir, exist_ok=True)",
                "    eval_score_path = eval_output_dir / config[\"eval_output\"]",
                "",
                "    # Get evaluation dataloader",
                "    _, _, eval_loader = get_loader(database_path, 1234, config)",
                "",
                "    print(\"Starting evaluation...\")",
                "    produce_evaluation_file(eval_loader, model, device, eval_score_path, eval_trial_path)",
                "    eval_eer, eval_tdcf = calculate_tDCF_EER(",
                "        cm_scores_file=eval_score_path,",
                "        asv_score_file=database_path / config[\"asv_score_path\"]",
                "    )",
                "    print(f\"\nEvaluation Results:\")",
                "    print(f\"  EER: {eval_eer:.3f}%\")",
                "    print(f\"  min t-DCF: {eval_tdcf:.5f}\")",
                "    print(\"✅ Evaluation complete!\")",
                "else:",
                "    print(\"⚠️ No best model (best.pth) found in Google Drive. Please train the model first.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "step8"},
            "source": ["## Step 8: Compare Results (Optional)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "compare_results"},
            "outputs": [],
            "source": [
                "# This step is optional and requires a baseline result to compare against.",
                "# You can manually input baseline EER and t-DCF values here.",
                "",
                "baseline_eer = 0.83  # Example baseline EER for AASIST on ASVspoof2019 LA eval",
                "baseline_tdcf = 0.0275 # Example baseline min t-DCF for AASIST on ASVspoof2019 LA eval",
                "",
                "# Assuming eval_eer and eval_tdcf are available from Step 7",
                "if 'eval_eer' in locals() and 'eval_tdcf' in locals():",
                "    print(f\"\nComparison with Baseline:\")",
                "    print(f\"  Your Model EER:      {eval_eer:.3f}% (Baseline: {baseline_eer:.3f}%)\")",
                "    print(f\"  Your Model min t-DCF: {eval_tdcf:.5f} (Baseline: {baseline_tdcf:.5f})\")",
                "",
                "    if eval_eer < baseline_eer:",
                "        print(\"  🎉 Your model achieved better EER than the baseline!\")",
                "    elif eval_eer == baseline_eer:",
                "        print(\"  Your model achieved similar EER to the baseline.\")",
                "    else:",
                "        print(\"  Your model EER is higher than the baseline.\")",
                "",
                "    if eval_tdcf < baseline_tdcf:",
                "        print(\"  🎉 Your model achieved better min t-DCF than the baseline!\")",
                "    elif eval_tdcf == baseline_tdcf:",
                "        print(\"  Your model achieved similar min t-DCF to the baseline.\")",
                "    else:",
                "        print(\"  Your model min t-DCF is higher than the baseline.\")",
                "else:",
                "    print(\"Please run Step 7 to evaluate the model before comparing results.\")"
            ]
        }
    ],
    "metadata": {
        "colab": {
            "collapsed_sections": [],
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open("/home/ubuntu/aasist/AASIST_AST_Colab.ipynb", "w") as f:
    json.dump(notebook_content, f, indent=4)

print("AASIST_AST_Colab.ipynb generated successfully!")
