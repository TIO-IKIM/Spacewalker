FROM nvcr.io/nvidia/tritonserver:23.09-py3 AS Spacewalker-triton

RUN python3 -m pip install --upgrade pip
RUN pip install torch accelerate open-clip-torch Pillow git+https://github.com/huggingface/transformers@v4.49.0-SigLIP-2