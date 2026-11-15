# Advisory records

These files preserve the research consultations and their inputs. They are working records rather than endorsed research claims. Some recommendations were corrected after the user's label clarification and subsequent technical review.

Read [the synthesis](../advisory_summary.md) for the accepted plan and rejected advice. The final task definition is normal-folder gait versus every other folder as abnormal gait.

- [Shared brief](brief.md) describes the evidence and requested critique. It was updated after the user's label clarification.
- [Claude first response](claude_review.md) records the initial recommendation. The original prompt is retained separately.
- [Claude revised response](claude_followup_review.md) records the second round with corrections and competing ideas.
- [GPT-6 Astra review](gpt_6_astra_review.md) develops the cross-joint and target-construction questions.
- [GPT-6 Sol review](gpt_6_sol_review.md) develops the evaluation and data-feasibility checks.
- [GAVD opportunity census](gavd_opportunity_audit.json) contains newly checked metadata and filename counts. It does not certify video decoding, participant identity, or pose quality.

The CLI invocation and response JSON files record the actual Claude calls. Raw prompts and responses are preserved for traceability, including advice that was later rejected. No new training results are contained in these consultations.

## Neurological and world-model extension

- [Claude's third consultation](claude_neuromechanics_review.md) critiques the proposed context profile and future physical model. Its prompt, response JSON, and invocation record are preserved with the same filename prefix.
- [GPT-6 Astra's neuromechanics review](neuromechanics_gpt_astra.md) checks prior simulation research, ambiguity, and experiment design.
- [GPT-6 Sol's feasibility review](neuromechanics_gpt_sol.md) maps proposed measurements to the actual 16-joint, 2D pose pipeline.
- [The clinical evidence ledger](neuromechanics_clinical_evidence.md), prepared by GPT-6.1 Sol, describes 14 primary studies and their measurement limits. The coordinator and a second clinical pass checked consequential claims used in the main document.

These are advisory records. The accepted [neurological research direction](../neurological_research_direction.md) and [roadmap](../research_roadmap.md) resolve conflicts and contain the final proposed protocol. The proposal has not been implemented or validated.
