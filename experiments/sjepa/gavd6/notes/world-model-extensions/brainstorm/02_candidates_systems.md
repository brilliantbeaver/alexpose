# Systems-first candidates

A **world model** predicts state change. **Latent prediction** forecasts internal vectors. **Inverse dynamics** recovers a cause from a transition. **SMPL** is a parametric 3D body. A **JEPA** predicts masked latent targets instead of coordinates.

Protocol: `SV5` is five folds over decodable GAVD sequences, grouped by source video, never subject. `P25` is day 1 frozen extraction, day 2 adapters, day 3 SV5 evidence/controls. One H100 per run: `C25 = 4` frozen-feature H100-hours `+ 25/100 x 3 hours x 3 arms x 3 seeds = 10.75 H100-hours`; parallelism reduces wall time. No hardware conversion.

Lens tags: `P` vision physics; `G` physical grounding; `J` discriminative/generative S-JEPA; `S` simulator or OpenSim loss; `D` depth on motion embeddings; `Q` Qwen RL tools; `O` body-object keypoints. `S0` matches duration, centroid drift, foreground area, and pose confidence.

Boundary: no GoalForce/MVA pixel-control clone, VideoMDM lifter training, PhyMotion/PP-Motion scalar-score clone, inaccessible SC3 reproduction, or clinical frozen-probe headline.

1. **Depth-gated surprise.** Q: Can depth uncertainty separate gait surprise from camera error? Bet: yes. Base `Depth-Anything-V2-Small-hf` plus local `outputs/repaired-jepa-seed7-v2/seed-7_standard_sjepa_best.pt`; add zero-init depth tokens. Bridge AMASS 3D camera/pose corruptions to GAVD 2D+depth. Evidence/null: anatomical-error stability versus confidence-only; no gain rejects depth rescue. Control `S0`+depth shuffle. Pilot `P25/C25`. `[D,J]`

2. **Lift-consensus abstention.** Q: Is lift disagreement uncertainty rather than pathology? Bet: yes. Freeze `wham_vit_w_3dpw.pth.tar` and `gvhmr_siga24_release.ckpt`; train a two-layer mixer. Bridge by projecting/re-lifting AMASS and ensembling GAVD lifts. Evidence/null: risk-coverage and repair agreement; universal disagreement forbids 3D claims. Control `S0`+known AMASS corruptions. Pilot `P25/C25`. `[G,D]`

3. **Residual-conditioned dynamics.** Q: Are OpenSim residual and normal-prior error separate axes? Bet: yes. Freeze `GaitDynamicsDiffusion.pt`, `GaitDynamicsRefinement.pt`, and local JEPA; add zero-init residual tokens. Bridge SMPL to Rajagopal, corrupted projection, then GAVD multi-lift retargeting. Evidence/null: two-dimensional error geometry; collinearity rejects a second axis. Control `S0`+speed-match/residual-shuffle. Pilot `P25/C25`. `[P,G,S]`

4. **Body-device world state.** Q: Do device motions improve prediction beyond body pose? Bet: yes. Freeze `sam2.1_hiera_large.pt` and local JEPA; add rank-8 object-token adapters. Bridge projected AMASS plus synthetic device tracks to SAM-tracked GAVD. Evidence/null: held-source body/device forecast likelihood; no gain makes devices context-only. Control `S0`+device-only/time-shuffle. Pilot `P25/C25`, with ProGait stress test. `[O,J]`

5. **Expected cut: tool agent.** Q: Can selective sensing beat equal-cost all-tools? Bet: yes. Freeze `Qwen/Qwen3-VL-8B-Instruct`, SAM2, Depth Anything, and WHAM; RL-train rank-8 Qwen tool LoRA. Bridge AMASS corruption episodes with known clean 3D to GAVD. Evidence/null: selective-risk/tool-cost curve; all-tools dominance rejects agency. Control `S0`+random equal-budget policy. Pilot `P25/C25`. `[Q,D,G]`

6. **Frozen-video latent adapter.** Q: Do video features stabilize skeleton prediction across appearance? Bet: yes. Freeze V-JEPA 2 `vitl.pt` and local JEPA; learn rank-8 cross-modal prediction tokens. Bridge randomized AMASS skeleton composites to paired GAVD RGB+pose. Evidence/null: future-latent consistency across views; raw 2D winning rejects the video prior. Control `S0`+background-only/person-mask swap. Pilot `P25/C25`. `[P,J]`

7. **Generative JEPA posterior.** Q: Do local JEPA latents support multiple futures? Bet: yes. Freeze `outputs/repaired-jepa-seed7-v2/seed-7_standard_sjepa_best.pt`; train only a low-rank mixture-density decoder. Bridge AMASS projection/occlusion noise to GAVD lift ensembles. Evidence/null: future coverage and anatomical calibration; random/raw parity rejects generative content. Control `S0`+speed-match/random encoder. Pilot `P25/C25`. `[J,G]`

8. **Consensus minimum repair.** Q: Is the smallest normalization stable across lifts? Bet: yes. Freeze `GaitDynamicsDiffusion.pt` and `saved_models_final/vae_final.pth`; train a two-layer SMPL-to-Rajagopal retargeter. Bridge AMASS metric SMPL to multi-lift GAVD. Evidence/null: repair-vector agreement; camera-following edits invalidate interpretation. Control `S0`+camera perturbation/speed match. Pilot `P25/C25`. `[G,S]`

9. **Expected cut: robot-action transplant.** Q: Do V-JEPA 2-AC dynamics survive a Core11 action mapper? Bet: unlikely. Freeze `vjepa2-ac-vitg.pt`; adapt mapper and inverse head. Bridge perturbed/projected AMASS to GAVD lifts. Evidence/null: forward-inverse cycle residual; failure shows embodiment specificity. Control `S0`+action time-shuffle. Pilot `P25/C25`; cut on day 2 unless held-out AMASS beats raw deltas. `[J,G]`
