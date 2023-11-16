# multi-modal_place_recognition

- system requirements

  ```
  Ubuntu
  CUDA>11.0 (11.8 is recommended)
  python 3.8
  ```

  torch installation (take CUDA 11.8 as the example)

  ```
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

  MinkowskiEngine installation

  ```
  conda install openblas-devel -c anaconda
  pip install pip==22.3.1
  pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
  ```

  sptr installation

  ```
  cd ~
  pip install torch_scatter==2.0.9
  pip install torch_geometric==1.7.2
  pip install torch_cluster
  pip install torch_sparse
  pip install timm
  git clone https://github.com/dvlab-research/SparseTransformer.git
  cd SparseTransformer
  python setup.py install
  ```

  spconv

  ```
  pip install spconv-cu118
  ```

  others

  ```
  pip install scikit-learn
  pip install tqdm
  pip install pytorch-metric-learning==1.1
  pip install tensorboard
  pip install tensorboardX
  pip install torchsummary
  pip install open3d
  ```

- datasets

  We provide the preprocessed  Boreas dataset in

  https://drive.google.com/file/d/1zWF8uSmnDgzYczuuoK-w-zF_AnVV7o95/view?usp=share_link

  After downloading, change the two arguments in tools/options.py, --dataset_folder and --image_path, as where you store your Boreas dataset.

- training

  You can run train.py to start training.
  You can run evaluate.py to start evaluating, where you need to configure the model weight path.

- training

  You can run train.py to start training.
  You can run evaluate.py to start evaluating, where you need to configure the model weight path.
