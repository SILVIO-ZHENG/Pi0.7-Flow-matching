# G1 π0.7-Inspired Reproduction Notes

These documents define the reproducible engineering contracts added around the OpenPI model path.

- [data_contract.md](data_contract.md): ROS2 messages, action-centred alignment, episode files, LeRobot fields, and 43/28/32-dimensional ordering.
- [joint_objective.md](joint_objective.md): hierarchical subtasks, FAST-token cross-entropy, continuous Flow Matching, and Knowledge Insulation.
- [training_time_rtc.md](training_time_rtc.md): asynchronous inference latency, training-time prefix conditioning, and rolling deployment replanning.
- [runbook.md](runbook.md): commands from ROS2 recording and episode labelling through LeRobot conversion, two-stage training, and dry-run deployment.

```text
XR -> IK/retargeting -> ROS2/MCAP -> aligned Parquet and MP4
   -> episode QC/split -> LeRobot V3 -> q01/q99
   -> PaliGemma FAST-CE + Flow Action Expert
   -> 10-step flow sampling -> RTC cache -> joint-limit gate
```

A complete official π0.7 implementation is not available in this source tree. The phrase “π0.7-inspired” identifies experimental integrations around the public OpenPI π0.5 path. Completed status must be supported by tests or experiment records.
