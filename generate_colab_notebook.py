import json

notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {"id": "title"},
            "source": [
                "# AASIST + AST: Audio Spoof Detection with Audio Spectrogram Transformer (Colab with Resume Support)\n",
                "\n",
                "This notebook trains the **AASIST-AST** hybrid model, combining AASIST with an Audio Spectrogram Transformer (AST) encoder. It includes **checkpointing and resume functionality** to overcome Google Colab's session limits.\n",
                "\n",
                "## Architecture Overview\n",
                "- **AASIST Branch**: Raw waveform → SincConv → ResNet encoder → Spectral & Temporal GATs → Graph pooling\n",
                "- **AST Branch**: Raw waveform → Mel Spectrogram → Patch Embedding → Transformer Encoder → CLS token\n",
                "- **Fusion**: Concatenation of both branches → Final binary classifier\n",
                "\n",
                "## Expected Performance\n",
                "The baseline AASIST achieves **EER: 0.83%, min t-DCF: 0.0275** on ASVspoof2019 LA eval set.  \n",
                "The AASIST-AST hybrid is expected to improve upon this by leveraging global spectro-temporal attention from the Transformer.\n",
                "\n",
                "---\n",
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
                "import torch\n",
                "print(f\"PyTorch version: {torch.__version__}\")\n",
                "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
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
                "from google.colab import drive\n",
                "drive.mount(\"/content/drive\")\n",
                "\n",
                "# Define a path in your Google Drive to save/load checkpoints\n",
                "DRIVE_CHECKPOINT_DIR = \"/content/drive/MyDrive/AASIST_AST_Checkpoints\"\n",
                "!mkdir -p {DRIVE_CHECKPOINT_DIR}\n",
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
                "import os\n",
                "\n",
                "# Change to /content/ directory to ensure correct cloning path\n",
                "%cd /content/\n",
                "\n",
                "# Clone the forked repository if it doesn't exist\n",
                "if not os.path.exists(\"aasist\"):\n",
                "    !git clone https://github.com/ahmadSh96/aasist.git\n",
                "\n",
                "# Change into the cloned repository directory\n",
                "%cd aasist\n",
                "\n",
                "# Switch to the AST integration branch\n",
                "!git checkout feature/ast-integration\n",
                "!git pull origin feature/ast-integration\n",
                "\n",
                "# Install dependencies\n",
                "!pip install -q torchcontrib soundfile torchaudio\n",
                "\n",
                "print(\"\\n✅ Setup complete!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "step4"},
            "source": [
                "## Step 4: Download ASVspoof 2019 LA Dataset\n",
                "\n",
                "> **Note**: The dataset is ~10GB. This will take several minutes.\n",
                "> If you already have the dataset in your Drive, it will be copied to the local runtime."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "download_data"},
            "outputs": [],
            "source": [
                "import os\n",
                "from google.colab import drive\n",
                "\n",
                "# 1. Ensure Drive is mounted\n",
                "if not os.path.exists('/content/drive'):\n",
                "    drive.mount('/content/drive')\n",
                "\n",
                "# Define paths\n",
                "drive_path = \"/content/drive/MyDrive/LA.zip\"\n",
                "local_zip_path = \"/content/LA.zip\"\n",
                "extract_path = \"/content/aasist/LA\"\n",
                "\n",
                "# 2. Check if the extracted folder already exists\n",
                "if not os.path.exists(extract_path):\n",
                "    # 3. Check if the zip file exists in Drive\n",
                "    if os.path.exists(drive_path):\n",
                "        print(\"📦 Found LA.zip in Google Drive. Copying to local runtime...\")\n",
                "        !cp \"{drive_path}\" \"{local_zip_path}\"\n",
                "    else:\n",
                "        print(\"🌐 LA.zip not found in Drive. Downloading (~10GB)...\")\n",
                "        !wget -q --show-progress https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip -O \"{local_zip_path}\"\n",
                "        \n",
                "        # Save a copy to Drive for future use\n",
                "        print(\"💾 Saving a copy to Google Drive for future use...\")\n",
                "        !cp \"{local_zip_path}\" \"{drive_path}\"\n",
                "\n",
                "    # 4. Extract the zip file\n",
                "    print(\"🔓 Extracting LA.zip...\")\n",
                "    !unzip -q \"{local_zip_path}\" -d /content/aasist\n",
                "    \n",
                "    # Remove local zip to save space\n",
                "    !rm \"{local_zip_path}\"\n",
                "    print(\"✅ Dataset ready!\")\n",
                "else:\n",
                "    print(\"✅ Dataset folder 'LA' already exists in local runtime, skipping.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "step5"},
            "source": [
                "## Step 5: Verify Model Architecture\n",
                "\n",
                "Let's inspect the AASIST-AST model to confirm it loads correctly and count parameters."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "verify_model"},
            "outputs": [],
            "source": [
                "import sys\n",
                "sys.path.insert(0, \"./\")\n",
                "\n",
                "import json\n",
                "import torch\n",
                "\n",
                "# Load config\n",
                "with open(\"config/AASIST_AST.conf\", \"r\") as f:\n",
                "    config = json.load(f)\n",
                "\n",
                "model_config = config[\"model_config\"]\n",
                "\n",
                "# Load model\n",
                "from models.AASIST_AST import Model\n",
                "model = Model(model_config)\n",
                "\n",
                "# Count parameters\n",
                "total_params = sum(p.numel() for p in model.parameters())\n",
                "trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
                "\n",
                "print(f\"Model: AASIST-AST\")\n",
                "print(f\"Total parameters:     {total_params:,}\")\n",
                "print(f\"Trainable parameters: {trainable_params:,}\")\n",
                "\n",
                "# Test forward pass\n",
                "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
                "model = model.to(device)\n",
                "dummy_input = torch.randn(2, 64600).to(device)  # batch=2, 4 seconds at 16kHz\n",
                "with torch.no_grad():\n",
                "    features, output = model(dummy_input)\n",
                "\n",
                "print(f\"\\nForward pass test:\")\n",
                "print(f\"  Input shape:    {dummy_input.shape}\")\n",
                "print(f\"  Features shape: {features.shape}\")\n",
                "print(f\"  Output shape:   {output.shape}\")\n",
                "print(\"\\n✅ Model loaded and forward pass successful!\")"
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
                "import os\n",
                "import glob\n",
                "import shutil\n",
                "\n",
                "# Define output directory for this Colab session\n",
                "COLAB_OUTPUT_DIR = \"colab_exp_result\"\n",
                "os.makedirs(COLAB_OUTPUT_DIR, exist_ok=True)\n",
                "\n",
                "print(f\"DRIVE_CHECKPOINT_DIR: {DRIVE_CHECKPOINT_DIR}\")\n",
                "print(\"Listing contents of DRIVE_CHECKPOINT_DIR:\")\n",
                "!ls -R {DRIVE_CHECKPOINT_DIR}\n",
                "\n",
                "# Find latest checkpoint in Google Drive\n",
                "latest_checkpoint = None\n",
                "checkpoint_files = sorted(glob.glob(f\"{DRIVE_CHECKPOINT_DIR}/**/weights/checkpoint.pth\", recursive=True), key=os.path.getmtime)\n",
                "print(f\"Found checkpoint files: {checkpoint_files}\")\n",
                "if checkpoint_files:\n",
                "    latest_checkpoint = checkpoint_files[-1]\n",
                "    print(f\"Found latest checkpoint: {latest_checkpoint}\")\n",
                "\n",
                "# Construct training command\n",
                "train_command_parts = [\n",
                "    \"python main.py\",\n",
                "    \"--config config/AASIST_AST.conf\",\n",
                "    f\"--output_dir {COLAB_OUTPUT_DIR}\",\n",
                "    \"--seed 1234\",\n",
                "    \"--pretrained_aasist_path models/weights/AASIST.pth\"\n",
                "]\n",
                "\n",
                "if latest_checkpoint:\n",
                "    train_command_parts.append(f\"--resume_checkpoint {latest_checkpoint}\")\n",
                "\n",
                "train_command = \" \".join(train_command_parts)\n",
                "\n",
                "print(\"Starting training with command:\")\n",
                "print(f\"!{train_command}\")\n",
                "!{train_command}\n",
                "\n",
                "# After training, copy results to Google Drive\n",
                "print(f\"\\nCopying results from {COLAB_OUTPUT_DIR} to {DRIVE_CHECKPOINT_DIR}...\")\n",
                "model_tag_dirs = glob.glob(f\"{COLAB_OUTPUT_DIR}/*_ep*_bs*\")\n",
                "if model_tag_dirs:\n",
                "    actual_output_to_copy = model_tag_dirs[0]\n",
                "    print(f\"Copying actual output directory: {actual_output_to_copy} to {DRIVE_CHECKPOINT_DIR}/\")\n",
                "    !cp -r {actual_output_to_copy}/* {DRIVE_CHECKPOINT_DIR}/\n",
                "    print(\"✅ Results copied to Google Drive!\")\n",
                "else:\n",
                "    print(\"❌ No output directory found to copy.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "step7"},
            "source": ["## Step 7: Evaluate Results"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "evaluate"},
            "outputs": [],
            "source": [
                "import os\n",
                "import glob\n",
                "\n",
                "print(\"--- Final Evaluation Results ---\")\n",
                "log_files = glob.glob(f\"{DRIVE_CHECKPOINT_DIR}/metric_log.txt\")\n",
                "if log_files:\n",
                "    with open(log_files[0], \"r\") as f:\n",
                "        print(f.read())\n",
                "else:\n",
                "    print(\"No metric log found yet. Training might still be in progress or failed.\")\n",
                "\n",
                "print(\"\\n--- EER and t-DCF Summary ---\")\n",
                "summary_files = glob.glob(f\"{DRIVE_CHECKPOINT_DIR}/t-DCF_EER.txt\")\n",
                "if summary_files:\n",
                "    with open(summary_files[0], \"r\") as f:\n",
                "        print(f.read())\n",
                "else:\n",
                "    print(\"No final summary found yet.\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("AASIST_AST_Colab.ipynb", "w") as f:
    json.dump(notebook_content, f, indent=1)

print("AASIST_AST_Colab.ipynb generated successfully!")
