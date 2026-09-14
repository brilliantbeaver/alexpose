"""Readable figures for the cache-only repair-mechanism diagnostic notebook."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROLE_COLORS = {"train": "#587743", "calibration": "#286b91", "development": "#c57927"}


def _event_and_noise(table):
    return table.loc[table.fixture.eq("factorial") & table.event_present.astype(bool)
                     & table.noise_present.astype(bool)].copy()


def _roles(table):
    return [role for role in ("train", "calibration", "development") if role in table.role.unique()]


def plot_repair(report):
    """Show absolute error beside event fidelity on the same difficult cases."""
    table = _event_and_noise(report["summary"])
    roles = _roles(table)
    methods = table.method.drop_duplicates().tolist()
    privileged = {"oracle_mixture_unprojected", "oracle_mixture_projected",
                  "raw_projected_reference_lengths", "truth_projected_observed_lengths",
                  "truth_projected_reference_lengths"}
    colors = ["#9099a1" if method in privileged else "#286b91" for method in methods]
    fig, axes = plt.subplots(len(roles), 2, squeeze=False,
                             figsize=(13, max(3.4, .33*len(methods)+.8)*len(roles)),
                             constrained_layout=True)
    for row, role in enumerate(roles):
        values = table.loc[table.role.eq(role)].set_index("method").reindex(methods)
        y = np.arange(len(methods))
        for ax, metric, scale, label in (
                (axes[row, 0], "observed_mse_m2", 1e4, "Observed joint squared error (cm²)"),
                (axes[row, 1], "retention", 1, "Event retention (1 is exact; unclipped)")):
            ax.barh(y, values[metric].to_numpy()*scale, color=colors, alpha=.85)
            ax.set(yticks=y, yticklabels=methods, xlabel=label,
                   title=f"{role.capitalize()}: event + tracking error")
            ax.invert_yaxis()
            ax.axvline(0, color="#86939c", lw=.8)
            ax.grid(axis="x", alpha=.15)
            if metric == "retention":
                ax.axvline(1, color="#087f78", lw=1, ls="--")
    fig.suptitle("Read repair and preservation together; these are diagnostic comparisons")
    fig.legend(handles=[Patch(color="#286b91", label="Permitted-input method"),
                        Patch(color="#9099a1", label="Reference-informed diagnostic")],
               loc="outside lower center", ncol=2, fontsize=9)
    return fig


def plot_strength_curves(report):
    """Display prespecified diagnostic strengths, without selecting a winner."""
    table = _event_and_noise(report["strength_curve"])
    roles = _roles(table)
    methods = table.method.drop_duplicates().tolist()
    colors = {method: plt.get_cmap("tab10")(i % 10) for i, method in enumerate(methods)}
    fig, axes = plt.subplots(len(roles), 2, squeeze=False,
                             figsize=(13, 3.6*len(roles)), constrained_layout=True)
    for row, role in enumerate(roles):
        for method, group in table.loc[table.role.eq(role)].groupby("method", sort=False):
            variants = group.groupby("projection", dropna=False) if "projection" in group else [(False, group)]
            for projection, values in variants:
                values = values.sort_values("strength")
                label = f"{method}; projection={projection}" if "projection" in group else method
                projected = projection is True or str(projection).lower() in {"true", "projected", "legacy", "fixed_lengths"}
                for ax, metric, scale in ((axes[row, 0], "observed_mse_m2", 1e4),
                                          (axes[row, 1], "retention", 1)):
                    ax.plot(values.strength, values[metric]*scale, marker="o", ms=3,
                            color=colors[method], ls="--" if projected else "-", label=label)
        for ax in axes[row]:
            ax.set_xscale("symlog", linthresh=.025)
            ax.set_xticks([0, .025, .1, .25, .5, 1], ["0", ".025", ".1", ".25", ".5", "1"])
            ax.set(xlabel="Repair strength (axis expanded near zero)", title=role.capitalize())
            ax.grid(alpha=.15)
        axes[row, 0].set_ylabel("Observed joint squared error (cm²)")
        axes[row, 1].set_ylabel("Event retention (unclipped)")
        axes[row, 1].axhline(1, color="#86939c", lw=.8, ls=":")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    fig.suptitle("Strength curves explain the mechanism; they do not retune the official result")
    return fig


def plot_bones(report):
    """Compare absolute length bias with variation in noisy factorial cases."""
    table = report["bone_lengths"]
    table = table.loc[table.fixture.eq("factorial") & table.noise_present.astype(bool)].copy()
    table["absolute_bias_m"] = table.bias_m.abs()
    metrics = ["absolute_bias_m", "reference_length_std_m"]
    people = table.groupby(["role", "person_id", "joint", "joint_name"], as_index=False)[metrics].mean()
    means = people.groupby(["role", "joint", "joint_name"], as_index=False)[metrics].mean()
    joints = means[["joint", "joint_name"]].drop_duplicates().sort_values("joint")
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, .30*len(joints))), constrained_layout=True)
    for role in _roles(means):
        values = means.loc[means.role.eq(role)].set_index("joint").reindex(joints.joint)
        for ax, metric in zip(axes, metrics):
            ax.plot(values[metric]*1000, np.arange(len(joints)), "o-", ms=4,
                    color=ROLE_COLORS[role], label=role)
    for ax in axes:
        ax.set(yticks=np.arange(len(joints)), yticklabels=joints.joint_name)
        ax.invert_yaxis()
        ax.axvline(0, color="#86939c", lw=.8)
        ax.grid(axis="x", alpha=.15)
        ax.legend()
    axes[0].set(xlabel="Mean absolute length bias (mm)", title="Magnitude without sign cancellation")
    axes[1].set(xlabel="Within-clip reference length standard deviation (mm)",
                title="Reference lengths can vary over time")
    fig.suptitle("Noisy factorial cases: average within each person, then average people")
    return fig


def plot_flow(report):
    """Show available-support paired evidence and its coverage, not certainty."""
    pairs, cases = report["flow_pairs"], report["flow_cases"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    valid = pairs.loc[pairs.status.eq("valid_available_support_pair")] if len(pairs) else pairs
    colors = ROLE_COLORS
    limits = []
    for role in _roles(valid) if len(valid) else []:
        group = valid.loc[valid.role.eq(role)]
        x, y = group.failure_event_transport_gap_mean_px, group.real_event_transport_gap_mean_px
        axes[0].scatter(x, y, label=role, color=colors[role], alpha=.8)
        limits.extend(x.tolist()+y.tolist())
    finite = np.asarray(limits, float)
    finite = finite[np.isfinite(finite)]
    if len(finite):
        lo, hi = min(0, finite.min()), max(0, finite.max())
        padding = max((hi-lo)*.1, .05)
        axes[0].plot([lo-padding, hi+padding], [lo-padding, hi+padding],
                     color="#86939c", ls="--", lw=1)
        axes[0].set(xlim=(lo-padding, hi+padding), ylim=(lo-padding, hi+padding))
        axes[0].legend()
    axes[0].axhline(0, color="#c2c9cd", lw=.8)
    axes[0].axvline(0, color="#c2c9cd", lw=.8)
    axes[0].set(xlabel="Tracking-failure video: prior error minus raw error (px)",
                ylabel="Real-event video: prior error minus raw error (px)",
                title=("Above diagonal: real-event video favors raw more\n"
                       f"{len(valid)} supported pairs; {len(pairs)-len(valid)} unresolved or unmatched"))
    matched = cases.loc[cases.fixture.eq("matched")] if len(cases) else cases
    for role in _roles(matched) if len(matched) else []:
        group = matched.loc[matched.role.eq(role)]
        for event, marker in ((True, "o"), (False, "x")):
            points = group.loc[group.event_present.astype(bool).eq(event)]
            axes[1].scatter(points.event_displacement_max_px, points.event_flow_coverage,
                            color=colors[role], marker=marker, alpha=.7,
                            label=f"{role}: {'real event' if event else 'tracking failure'}")
    axes[1].set(xlabel="Maximum clean-to-edited projected separation (px)",
                ylabel="Usable flow / event transition locations", ylim=(-.03, 1.03),
                title="Geometric event size and available flow coverage")
    if len(matched):
        axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle("Available-support contrast: the two videos can contribute different locations")
    return fig


def plot_trace(trace):
    """Keep the sign of foot-height errors; show affected-joint error separately."""
    series = trace["series"]
    times = np.asarray(trace["timestamps"])
    truth = np.asarray(series["truth"])
    observed = np.asarray(trace["observed"], bool)
    affected = np.asarray(trace["affected_mask"], bool) & observed
    names = ("truth", "clean", "event_reference", "raw", "projected_raw", "prior", "bridge")
    colors = {"truth": "#087f78", "clean": "#697782", "event_reference": "#67ad89",
              "raw": "#c57927", "projected_raw": "#ad4b3c", "prior": "#286b91", "bridge": "#8864a3"}
    fig, grid = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    axes = grid.ravel()
    for name in names:
        if name not in series:
            continue
        value = np.asarray(series[name])
        style = ":" if name in {"clean", "event_reference"} else "-"
        axes[0].plot(times, value[:, 10, 1]*100, color=colors[name], ls=style, label=name)
        if name in {"truth", "clean", "event_reference"}:
            continue
        axes[1].plot(times, (value[:, 10, 1]-truth[:, 10, 1])*100,
                     color=colors[name], label=name)
        squared = np.sum((value-truth)**2, axis=-1)
        for ax, mask in zip(axes[2:], (affected, observed & ~affected)):
            counts = mask.sum(axis=1)
            error = np.sqrt(np.divide((squared*mask).sum(axis=1), counts,
                                      out=np.full(len(times), np.nan), where=counts > 0))*100
            ax.plot(times, error, color=colors[name], label=name)
    axes[0].set(ylabel="Left foot height (cm)", title="World vertical coordinate")
    axes[1].set(ylabel="Output minus reference height (cm)", title="Positive means too high")
    axes[2].set(ylabel="Position RMSE (cm)", title="Originally corrupted observed joints")
    axes[3].set(ylabel="Position RMSE (cm)", title="Previously accurate observed joints")
    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=.15)
    axes[1].axhline(0, color="#86939c", lw=.8)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"{trace['case_id']} | {trace['event_family']}", fontsize=10)
    return fig
