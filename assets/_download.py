from huggingface_hub import snapshot_download

snapshot_download(repo_id="TianxingChen/RoboTwin-CVPR-Challenge-2025", 
                  local_dir='.', 
                  repo_type="dataset",
                  resume_download=True)