# Inference architecture

`rrdbnet_arch.py` adapts the RRDBNet architecture from
[BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py),
copyright 2018–2022 BasicSR Authors, under Apache-2.0. The complete license is in
`BasicSR_LICENSE.txt`. Changes remove the registry dependency and provide local
initialization, layer construction and pixel-unshuffle helpers. All inference
parameters are loaded strictly from the official pretrained checkpoint.

The weights are not vendored. `run_sharpness_trials.ps1` downloads official
[Real-ESRGAN x2plus](https://github.com/xinntao/Real-ESRGAN) release weights into
temporary storage and verifies the recorded SHA-256. Consult the upstream
project for its license and model terms. No model was trained in this pilot.
