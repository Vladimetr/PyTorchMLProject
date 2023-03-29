docker run -d -p 3500:5000 \
    -v /mnt/nvme/vovik/tutorial/torch_project/dev/mlflow/data:/mlflow/mlruns/ \
    --user 1018:1018 \
    --name mlflow-server  \
     burakince/mlflow \
    mlflow server --backend-store-uri /mlflow/mlruns/ --default-artifact-root /mlflow/mlruns/ --host=0.0.0.0 --port=5000
