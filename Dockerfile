FROM nvcr.io/nvidia/pytorch:26.08-py3
COPY requirements.txt security-requirements.txt /opt/persona-image-lab/
RUN python -m pip install --no-cache-dir --ignore-installed --no-deps \
        -r /opt/persona-image-lab/security-requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/persona-image-lab/requirements.txt
ENV HF_HOME=/workspace/cache/huggingface PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 GRADIO_ANALYTICS_ENABLED=False \
    HF_HUB_DISABLE_TELEMETRY=1 PERSONA_HOST=0.0.0.0
WORKDIR /workspace
COPY lab.py history.py manage.py settings.py persona_profiles.py benchmark_batches.py \
     demos.json VERSION /workspace/
COPY tests /workspace/tests
CMD ["python", "lab.py", "--serve"]
