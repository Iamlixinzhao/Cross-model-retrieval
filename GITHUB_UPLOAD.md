# Uploading this project to GitHub

## 1. .gitignore

The repo `.gitignore` is set up so that **generated data is not pushed**, including:

- **Embeddings**: `emb_text.pt`, `emb_video.pt`, `*_embeddings/`, `msrvtt_*/`, `cheb_export_cim/`, `embeddings.npz`
- **Checkpoints**: `*.pth`, `*.pt`, `ckpt_cheb_gated/`, `pcme_checkpoints/`, etc.
- **Run output**: `sweep_runs/`, `results/`, `*.npz`, large JSON result files (optional)
- **Datasets**: `ImageBind/`, `data/`, `datasets/`
- **Environment**: `venv/`, `__pycache__/`, IDE and OS junk

Only code, configs, READMEs, and small metadata are intended to be versioned. Regenerate embeddings and checkpoints locally after cloning (see README_CHEB.md).

---

## 2. Log in with a Personal Access Token (PAT)

GitHub no longer accepts account passwords for `git push`. Use a **Personal Access Token** instead.

1. On GitHub: **Settings → Developer settings → Personal access tokens → Tokens (classic)**.
2. **Generate new token (classic)**. Give it a name, choose an expiration, and enable at least **repo**.
3. Copy the token once (it won’t be shown again).

When you push, use the token as the password:

- **HTTPS**:  
  `git push https://YOUR_USERNAME@github.com/YOUR_USERNAME/REPO_NAME.git main`  
  When prompted for password, paste the **token** (not your GitHub password).

- **Or store it so you don’t type it every time:**  
  `git config --global credential.helper store`  
  Then the first successful push will save the token (use only on a machine you control).

---

## 3. Create the repo and push

On GitHub, create a **new repository** (no README/license if you already have them locally).

Then in this project directory:

```bash
cd /path/to/Cross-model-retrieval

# If you haven't initialized yet
git init
git add .
git commit -m "Initial commit: Chebyshev projector and benchmarks"

# Add your GitHub repo as remote (replace USER and REPO_NAME)
git remote add origin https://github.com/USER/REPO_NAME.git

# Push (use token as password when prompted)
git branch -M main
git push -u origin main
```

If the repo already had a remote:

```bash
git remote set-url origin https://github.com/USER/REPO_NAME.git
git push -u origin main
```

After this, only files not listed in `.gitignore` (code, docs, etc.) will be on GitHub; embeddings and other generated data stay local.
