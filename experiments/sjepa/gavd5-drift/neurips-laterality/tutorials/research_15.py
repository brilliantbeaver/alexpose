"""Editable tutorial: motion weighting, source fidelity, and noisy observations."""
from nbformat.v4 import new_notebook
from .masking_shared import md, code, setup_cell


def build_notebook():
    return new_notebook(cells=[
        md(r'''
        # 15 — Motion weighting: what should become harder to observe?

        Notebook 12's completed scattered-mask comparison did not establish a
        learned-over-initial movement advantage. The [tutorial](docs/TUTORIAL.md)
        identifies motion weighting as the missing S-JEPA comparator. We first
        check what that sampler does, including its response to tracking jumps.
        All coordinates below are generated. This notebook needs no recordings
        and does not train a model. Continue with [16](16_structured_masking_and_context.ipynb)
        for geometry, [17](17_motion_and_structure_pretraining.ipynb) for training,
        and [18](18_motion_information_and_readout.ipynb) for evaluation.

        ## 1. Read the method precisely

        [MAMP, ICCV 2023, §3.4](https://arxiv.org/html/2308.07092) uses displacement
        across one token length to score motion. A softmax gives sampling weights;
        adding Gumbel noise and taking the largest values samples hidden targets
        without replacement. [S-JEPA, ECCV 2024, §3](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf)
        adopts motion-based selection while predicting teacher features.

        For four-step tokens, the MAMP code computes mean absolute displacement
        across the four offsets and three axes, copies the second block's score
        into the first, and divides by the clip maximum before temperature
        scaling. That normalization appears in the
        [official implementation](https://github.com/maoyunyao/MAMP/blob/main/model_mamp/transformer.py),
        whereas the paper writes an unnormalized intensity softmax. Our
        `mamp_motion` follows the code on fully observed inputs. It adds explicit
        valid-transition handling for pose data. The temperature 0.8 is a declared
        starting setting, not a claim of optimality for gait.

        Write the log weight as $a_i=I_i/(\max_j I_j\,\tau+10^{-10})$.
        Draw $u_i\sim U(0,1)$, form $g_i=-\log(-\log u_i)$, and hide
        the $K$ largest $a_i+g_i$. The softmax normalization is a constant
        across candidates and therefore cancels from the ranking. Sampling
        weights are **not** the final inclusion probabilities when $K>1$.
        ''') ,
        setup_cell(),
        code('''
        time = np.arange(32, dtype=float)
        xyz = np.zeros((32, 33, 3))
        xyz[:, 27, 0] = 0.10 * time
        xyz[:, 28, 0] = 0.02 * time
        valid = np.ones((8, 33), dtype=bool)
        arms = study_arms("motion", blocks=8)
        logits, metadata = mamp_logits(xyz, valid)
        display(pd.Series(metadata, name="MAMP code convention"))
        ''') ,
        md('''
        ## 2. Compare motion rules under the same hidden count

        The existing `robust_motion` adaptation uses median Euclidean speeds
        across valid transitions, caps positive scores at their 95th percentile,
        and mixes 75% motion probability with 25% uniform probability. Speeds
        here are per prepared step because the training input has been resized;
        original seconds and metric distances are not available in this array.

        Both motion rules preferentially hide moving tokens. Every valid token
        remains eligible, including the slower leg. A uniform component protects
        coverage, but does not guarantee a target on each side in every draw.
        The plot estimates inclusion frequencies through repeated sampling.
        ''') ,
        code('''
        draws, hidden_count = 500, 32
        rates = {}
        for name, arm in arms.items():
            rng = np.random.default_rng(1501)
            counts = np.zeros_like(valid, dtype=float)
            for _ in range(draws):
                result = sample_study_mask(xyz, valid, arm, hidden_count, rng)
                assert result.mask.sum() == hidden_count
                counts += result.mask
            rates[name] = counts / draws
        display(pd.DataFrame({name: values.mean(axis=0)[[27, 28, 0]]
                              for name, values in rates.items()},
                             index=["faster left ankle", "slower right ankle", "stationary nose"]))
        fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
        for name, values in rates.items():
            ax.plot(values.mean(axis=0), label=name.replace("_", " "), marker=".")
        ax.set(xlabel="Landmark identity", ylabel="Estimated inclusion probability",
               title="Generated movement; 32 of 264 tokens hidden per draw")
        ax.legend()
        display(fig); plt.close(fig)
        ''') ,
        md('''
        ## 3. A pose jump can distort motion priorities

        An isolated coordinate jump raises the absolute-displacement score.
        Maximum normalization can then compress the contrast between legitimate
        motion and stationary regions elsewhere in the clip. The median-based
        adaptation is designed to reduce this effect. Compare the scores below;
        this constructed failure mode does not establish which rule wins on GAVD.
        ''') ,
        code('''
        noisy = xyz.copy()
        noisy[15, 15, 0] = 100.0
        rows = []
        for label, coordinates in (("clean", xyz), ("one wrist jump", noisy)):
            weights, _ = mamp_logits(coordinates, valid)
            robust, _ = motion_scores(coordinates, valid)
            rows.append({"input": label, "MAMP ankle log weight": weights[:, 27].mean(),
                         "MAMP wrist maximum": weights[:, 15].max(),
                         "robust ankle score": robust[:, 27].mean(),
                         "robust wrist maximum": robust[:, 15].max()})
        display(pd.DataFrame(rows))
        stationary, fallback = mamp_logits(np.zeros_like(xyz), valid)
        assert np.all(stationary == 0) and fallback["motion_fallback_uniform"]
        missing = valid.copy(); missing[3, 15] = False
        observed = np.repeat(missing, 4, axis=0)
        unavailable = noisy.copy(); unavailable[~observed] = np.nan
        result = sample_study_mask(unavailable, missing, arms["mamp_motion"], 32,
                                   np.random.default_rng(15), observation_valid=observed)
        assert not (result.mask & ~missing).any()
        print("Stationary fallback and exclusion of missing targets passed.")
        ''') ,
        md('''
        ## 4. What would make this useful evidence?

        Keep feature targets, teacher updates, loss weights, augmentation and
        encoder architecture fixed while changing the sampler. Report trained
        and initial features with the same readout, together with direct-pose
        controls. A lower loss against an arm's own teacher is insufficient.

        Ordinary masked pretraining may use the full observed training clip to
        choose its targets. The selected locations can themselves convey motion
        information; our fixed-mask content-isolation test does not exclude that
        channel. A future-only input requires motion computation from its observed
        prefix. Future prediction remains the separate task in Notebook 14.

        See the [source review and experiment specification](docs/MOTION_STRUCTURED_MASKING.md)
        for the exact adaptations, remaining anatomical choices, and next tests.
        ''')
    ])
