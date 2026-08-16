"""Guards the one environment property this project cannot detect at runtime.

There is no NVIDIA GPU on the target machine, and a CUDA torch build costs
roughly 3.5GB of image for nothing. The CPU build is not a preference here, it
is the difference between a container that deploys and one that does not.

This is an environment assertion rather than a unit test on purpose: it fails
when someone's `pip install` resolved differently from the documented one,
which is exactly the failure that has no other symptom until the image is built.
"""

import torch


def test_torch_is_a_cpu_build():
    assert torch.__version__.endswith("+cpu"), (
        f"torch is {torch.__version__}, not a CPU build. Reinstall with:\n"
        "  pip install -r requirements-dev.txt -c constraints.txt "
        "--extra-index-url https://download.pytorch.org/whl/cpu"
    )


def test_torch_does_not_see_a_cuda_device():
    # Belt and braces: a CPU wheel cannot report CUDA availability, so a True
    # here means the wheel is not what the version string claims.
    assert not torch.cuda.is_available()
