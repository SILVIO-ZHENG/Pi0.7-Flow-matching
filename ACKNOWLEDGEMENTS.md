# Acknowledgements and Research References

This repository depends on open-source libraries, public model releases, and published research. Their availability made this engineering reproduction possible. Inclusion below is attribution, not an endorsement or a claim of official affiliation.

## Core model and robotics software

- [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi): the model, training, normalization, checkpoint, and policy-serving foundation used by this repository.
- [π0: A Vision-Language-Action Flow Model for General Robot Control](https://www.physicalintelligence.company/blog/pi0): the flow-based VLA design underlying the continuous Action Expert path.
- [π0.5](https://www.physicalintelligence.company/blog/pi05): the public OpenPI model path used as the executable basis for this reproduction.
- [FAST action tokenization](https://www.physicalintelligence.company/research/fast): the action-token representation used by the auxiliary teacher-forcing objective.
- [Knowledge Insulation](https://www.physicalintelligence.company/research/knowledge_insulation): the Stop-Gradient motivation used to isolate the VLM from the continuous-action loss.
- [OpenVLA](https://openvla.github.io/) and the [official OpenVLA repository](https://github.com/openvla/openvla): an important open-source reference for VLA data contracts, fine-tuning, and robot-policy deployment.
- [Hugging Face LeRobot](https://github.com/huggingface/lerobot): dataset storage, episode structure, and robot-learning tooling.
- [PaliGemma](https://ai.google.dev/gemma/docs/paligemma): the vision-language backbone family used by OpenPI.
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747): the continuous vector-field training framework used by the Action Expert.
- [ROS 2](https://docs.ros.org/), [MoveIt 2](https://moveit.picknik.ai/), and [MCAP](https://mcap.dev/): robot messaging, inverse kinematics, recording, and replay infrastructure.
- [Unitree Robotics SDK2](https://github.com/unitreerobotics/unitree_sdk2): the intended external hardware interface; the production SDK bridge is not included in this repository.

## Source snapshot

The selective migration baseline is [Knight1112D/CBC_Pi0.7_Openpi](https://github.com/Knight1112D/CBC_Pi0.7_Openpi) at revision `4e9878feefd3192d2395443135651021b1832df3`. Original OpenPI documentation is retained in `UPSTREAM_OPENPI_README.md`.

## Citation and licensing guidance

Use the official citation entries supplied by each upstream paper or repository when publishing results. Review `LICENSE`, `LICENSE_GEMMA.txt`, `NOTICE`, individual source headers, and each model checkpoint's terms before redistribution or deployment.
