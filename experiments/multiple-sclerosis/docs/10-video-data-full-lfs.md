# Full gait-video dataset

## Purpose

`video-data-full/` contains the source MP4 clips used by the multiple-sclerosis gait experiments. The clips are organized by cohort:

| Directory | Cohort |
| --- | --- |
| `video-data-full/MS/` | Multiple sclerosis |
| `video-data-full/PD/` | Parkinson's disease |
| `video-data-full/Normal/` | Control / normal gait |

There are 91 clips in the current dataset. The directory is about 371 MB in a fully hydrated working tree.

## Storage policy

The files in this directory are managed with Git Large File Storage (Git LFS). The repository stores small pointer files in Git; the binary MP4 content is held by the configured LFS remote. The rule is deliberately scoped to `video-data-full/**` in `.gitattributes`, so it does not change how other media elsewhere in the repository are handled.

The repository's existing `*.mp4` ignore rule remains useful for avoiding accidental video additions. Files in this curated dataset are intentionally force-added when introduced and are then covered by the LFS rule.

## Getting a usable checkout

Install Git LFS once on a new machine, then fetch the dataset after cloning:

```bash
git lfs install
git clone <repository-url>
cd alexpose
git lfs pull
```

To obtain only this experiment's objects in a clone that has already been made:

```bash
git lfs pull --include="experiments/multiple-sclerosis/video-data-full/**"
```

`git lfs ls-files` lists the tracked clips. `git lfs fsck` checks the local LFS object store when diagnosing a partial or corrupted download.

## Adding or replacing clips

Before opening a pull request, confirm that an MP4 is represented by an LFS pointer and that its LFS object will be uploaded:

```bash
git add -f experiments/multiple-sclerosis/video-data-full/<cohort>/<clip>.mp4
git lfs ls-files
git lfs fsck
git push
```

Do not add `.DS_Store` files or derived analysis outputs to this directory. Keep video filenames stable when possible: notebooks and annotation tables may refer to them directly. If a replacement is necessary, describe the source, cohort, and reason for the update in the pull request.

## Review checklist

- Confirm the file is in the appropriate cohort directory.
- Confirm `git lfs ls-files` reports the clip.
- Confirm the pull request does not include a raw binary blob or macOS metadata.
- Confirm the LFS transfer succeeds before merging, so a fresh clone can hydrate the data.
