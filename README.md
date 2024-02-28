# VAEpreview

Fast, small (1MB size) and bad-quality VAE decoder for streaming.

based on https://huggingface.co/graphnull/vae_onnx

<img src="example.jpg" width="256" height="256">  |  <img src="example1v.jpg" width="256" height="256">

<img src="example2.jpg" width="256" height="256">  |  <img src="example2v.jpg" width="256" height="256">

##Run
import onnxruntime

vae_encoder = onnxruntime.InferenceSession('../VAEpreview.onnx')
vae_encoder.run(None, {"latent": latent_tensor })