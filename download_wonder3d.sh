mkdir threestudio/utils/wonder3D/ckpts
mkdir threestudio/utils/wonder3D/ckpts/unet
wget https://huggingface.co/camenduru/Wonder3D/tree/main/random_states_0.pkl -P threestudio/utils/wonder3D/ckpts
wget https://huggingface.co/camenduru/Wonder3D/tree/main/scaler.pt -P threestudio/utils/wonder3D/ckpts
wget https://huggingface.co/camenduru/Wonder3D/tree/main/scheduler.bin -P threestudio/utils/wonder3D/ckpts
wget https://huggingface.co/camenduru/Wonder3D/tree/main/unet/diffusion_pytorch_model.bin -P threestudio/utils/wonder3D/ckpts/unet
wget https://huggingface.co/camenduru/Wonder3D/tree/main/unet/config.json -P threestudio/utils/wonder3D/ckpts/unet
