# Cross-Modal Retrieval

This repository provides code and scripts to run cross-modal retrieval experiments (e.g., text-to-video, video-to-text) with different embedding dimensions, precisions, and candidate sizes. The setup is based on **Conda** for environment management and **pip** for dependency installation.

---

## Environment Setup

We recommend creating a dedicated Conda environment and installing dependencies with `pip`.

### 1. Create Conda Environment

```bash
conda create -n crossmodal_exp python=3.9 -y
2. Activate Environment
bash
Copy code
conda activate crossmodal_exp
3. Install Dependencies
GPU version (CUDA 11.8):

bash
Copy code
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
CPU version (if no GPU available):

bash
Copy code
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
Install other dependencies:

bash
Copy code
pip install faiss-gpu  # or faiss-cpu if no GPU
pip install numpy scipy scikit-learn matplotlib seaborn pandas tqdm h5py pillow
Requirements File
You can also install dependencies directly from requirements.txt:

txt
Copy code
torch>=1.13.0
torchvision>=0.14.0
torchaudio>=0.13.0
faiss-gpu>=1.7.2
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
tqdm>=4.62.0
h5py>=3.6.0
pillow>=8.3.0
Install via:

bash
Copy code
pip install -r requirements.txt
Verification
After installation, run the following to verify everything is working:

bash
Copy code
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import faiss; print('FAISS version:', faiss.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
Running Experiments
The workflow consists of three main steps: dataset preparation, experiment execution, and result analysis.
A job submission script (submit_job.sh) is included for running on an HPC cluster with GPU support.

1. Create COCO Dataset Embeddings
bash
Copy code
python coco_dataset_solution.py
This will generate CLIP embeddings for a subset of the COCO validation dataset and save them in ./coco_data/.

2. Run Cross-Modal Retrieval Experiments
bash
Copy code
python run_experiments.py \
    --data_path ./coco_data/coco_embeddings.npz \
    --output_dir ./results \
    --embedding_dims 256 512 1024 \
    --precisions fp16 int8 int4 \
    --candidate_sizes 32 100 1000 \
    --recall_k 1 5 10 \
    --batch_size 128 \
    --num_workers 4
This will test different embedding dimensions, quantization precisions, and candidate sizes, and store results in the ./results/ directory.

3. Analyze Results
bash
Copy code
python analyze_results.py \
    --results_dir ./results \
    --output_path ./results/experiment_summary.json
This step aggregates all experiment outputs into a single JSON summary with metrics such as average Recall@K, MRR, speed, and memory usage.

Using submit_job.sh (HPC Cluster)
If you are running on a cluster with a job scheduler (e.g., SGE, Slurm), you can use the provided submit_job.sh script:

bash
Copy code
qsub submit_job.sh
The script will:

Activate the Conda environment

Install dependencies (if missing)

Run dataset preparation, experiments, and analysis

Save results and create a backup copy in ~/backup_results/

Notes
Use the GPU version of PyTorch and FAISS if you have CUDA-capable hardware.

If running on a CPU-only machine, install faiss-cpu instead of faiss-gpu and use the CPU build of PyTorch.

For reproducibility, keep a copy of your requirements.txt under version control.

To improve performance on clusters without internet access, pre-download the HuggingFace CLIP model on a login node to populate your cache (~/.cache/huggingface).
