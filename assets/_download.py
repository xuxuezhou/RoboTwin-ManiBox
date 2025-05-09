from huggingface_hub import snapshot_download

snapshot_download(repo_id="ZanxinChen/RoboTwin_Challenge_Round2", 
                  local_dir='.', 
                  repo_type="dataset",
                  resume_download=True)