# Setup and Operation

## Supported Baseline

- ARM64 Linux with NVIDIA GB10 and 128 GB unified memory.
- Target: NVIDIA DGX Spark systems built on GB10.
- Working host NVIDIA driver and NVIDIA Container Toolkit.
- Docker Engine and Compose plugin 2.30+ (the service uses `gpus: all`).
- For the recommended prompt-enhancer path: host Python with vLLM and `ninja`.
  Validation on GB10 uses vLLM 0.27.1. If unavailable, the application can
  fall back to the slower Transformers implementation inside the Lab container.
- Internet for the container build and one-time model download.
- At least 140 GiB free initially when both prompt enhancers are enabled; outputs
  and reference copies continue to grow.

Persona Image Lab has been validated on driver 580.178.04. The upstream
benchmark baseline used 580.173.02. The base image is
`nvcr.io/nvidia/pytorch:26.08-py3`; the NVIDIA container supplies PyTorch/CUDA.
Do not replace it with an x86 container or an arbitrary PyTorch wheel.
See NVIDIA's [DGX Spark documentation](https://docs.nvidia.com/dgx/dgx-spark/)
and [Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
for host setup. This project does not install or change host drivers.

## Install

Clone on the Spark, then run the commands in the main README. `./persona build`
builds a local image; it does not download model weights. `./persona doctor` checks
architecture, CUDA, GPU identity, output permissions, and model file presence.
`./persona download --accept-model-license` downloads the pinned model revision
only after explicit acceptance. Review the model's terms before using the flag.
`./persona download-enhancers --accept-model-license` separately downloads the
pinned PE-T2I and PE-I2I revisions under the same explicit-license workflow.

The launcher runs containers with your host UID/GID to avoid root-owned output
files. Run it as your ordinary user, not through `sudo`, after configuring
Docker access on the host. A root-owned model/output directory from an earlier
manual install may need its ownership corrected by its administrator.

Docker builds use an allowlisted context containing the application source,
tests, version metadata, Dockerfile, and requirements needed by the image.
Model weights, prompts, generated images, credentials, private persona data,
and `.git` never enter the build context. The application checkout is not
mounted into the running container.

Private persona data is mounted separately at `/persona-data` and is read-only
to the application. The default host directory is `./data/personas`, which is
Git-ignored. Set `PERSONA_DATA_DIR` in `.env` to keep real persona files
outside the checkout. Writable mounts are limited to `model/`, `outputs/`, and
`cache/`. The launcher rejects `PERSONA_DATA_DIR` if it overlaps any of those
writable directories.

## Start, Stop, and Upgrade

```bash
./persona start
./persona status
./persona logs
./persona stop
```

Start performs preflight before launching. When prompt enhancement is enabled and
host vLLM is available, the launcher also starts PE-T2I and PE-I2I as FP8 vLLM
processes. They listen only on Unix-domain sockets under `cache/prompt-enhancer/`;
no additional TCP ports are exposed. `./persona status` reports each enhancer as
loading, ready, or stopped. `./persona stop` terminates both host PE processes as
well as the Docker service.

The accelerated PE services use `fp8_per_tensor` in eager mode. On the validated
GB10 system this improved text expansion throughput while avoiding the long
compile/CUDAGraph startup cost. If host vLLM or ninja is unavailable in the default
`auto` mode, the launcher selects the in-container Transformers fallback instead.

Before upgrading, stop the service and back up `outputs/`, `.env`, and any local
code changes. Update to a reviewed commit/tag, rebuild, run tests, and restart:

```bash
./persona stop
git pull --ff-only
./persona build
./persona test
./persona start
```

Rollback: stop, check out the previously recorded tag/commit, rebuild, restart.
Never delete `outputs/` or `model/` as part of an upgrade. Version 1 history is
additive and legacy per-image JSON records remain readable without modification.

## Networking and Configuration

The Docker host publishes `127.0.0.1:7862`, not a LAN interface. For machines on
the same Tailscale tailnet, the recommended remote-access path is Tailscale
Serve:

```bash
tailscale serve --bg 7862
tailscale serve status
```

Keep the Serve configuration after stopping Persona Image Lab; it does not load
the image model. Starting the application again restores the same tailnet-only
HTTPS endpoint. Do not use Tailscale Funnel, Gradio public sharing, or publish
the Docker port on `0.0.0.0` for this alpha.

SSH local port forwarding remains a fallback when Tailscale Serve is unavailable:

```bash
ssh -N -L 7862:127.0.0.1:7862 YOUR_SPARK_SSH_ALIAS
```

If port 7862 is occupied, copy `.env.example` to `.env` and choose a different
`PERSONA_HTTP_PORT`, then use that port in the browser and remote-access proxy. For example,
`ssh -N -L 7863:127.0.0.1:7863 YOUR_SPARK_SSH_ALIAS` forwards port 7863.

Advanced direct-Python settings: `PERSONA_MODEL_DIR`, `PERSONA_OUTPUT_DIR`,
`PERSONA_HOST` (defaults to loopback), and `PERSONA_PORT` (7860). Compose configures
the container to listen on its internal interface and restricts access at the
host. Keep the default model revision; replacing model files manually can make
generation provenance inaccurate and is outside this release's support scope.

`.env` also accepts `PERSONA_PROMPT_ENHANCER=0` to disable PE-T2I/PE-I2I,
`PERSONA_PE_BACKEND=auto|vllm|transformers` to control the runtime (`auto` is the
default), and `PERSONA_THEME=soft|ocean|monochrome|glass` to select a built-in Gradio theme.
Ocean is the reviewed default. Community theme IDs are accepted for local
experimentation, but they require Hub access at startup and are not the default.
