# Robot Brains Toybox

A [marimo](https://marimo.io) notebook of nine CPU-sized toys for getting acquainted with robot foundation models:

1. Two clocks: why System 2 and System 1 split
2. Flow matching vs MSE for multimodal actions
3. Real-time chunking vs naive switching and temporal ensembling
4. FAST action tokens (DCT, scale, round)
5. Metadata prompting and classifier-free guidance
6. VLA vs world action model (video model × inverse dynamics), with a FLARE-style latent-alignment head
7. In-context imitation and few-step fine-tuning
8. Federated learning across a robot fleet (FedAvg, FedProx, shared EEF space, DP)
9. A System 2 agent that learns skills, not weights

Open in molab (cloud, works in a phone browser):
https://molab.marimo.io/github/TheJDen/artifacts/blob/robot-brains-toys/robot_brains_toys.py

Or run locally:

```
uvx marimo edit --sandbox robot_brains_toys.py
```

Everything trains on CPU; the whole notebook takes about two minutes on two cores.
