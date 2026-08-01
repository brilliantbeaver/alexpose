# Tutorial: MLP Latent Features for 33-Keypoint Gait Sequences

## Short Recommendation

Transforming per-frame 33-keypoint poses into a learned latent feature space with a multi-layer perceptron can be a good idea, but it should not be the first model you trust.

For this project, start with your current 82 engineered gait features and classical classifiers as the baseline. Then add an MLP encoder only if it improves held-out, subject- or video-grouped validation. A learned latent space is useful when you have enough labeled sequences, consistent pose quality, and gait patterns that are not fully captured by hand-engineered summary features. It is risky when the dataset is small, labels are noisy, or train/test splits leak frames from the same video or subject.

Use this rule of thumb:

| Situation | Recommendation |
| --- | --- |
| Fewer than 100-200 labeled sequences total | Usually not worth it. Use engineered features, SVM, RF, XGBoost, or logistic regression. |
| 200-1000 labeled sequences | Try a small MLP encoder with strong regularization and grouped cross-validation. |
| 1000+ labeled sequences | A learned encoder becomes much more reasonable. |
| Many frames per video but few videos | Be careful. More frames are not the same as more independent training samples. |
| Need clinical interpretability | Keep engineered features as the primary model and use the latent model as an experimental comparison. |

## What Problem Are We Solving?

You currently have a sequence of frames, and each frame contains 33 pose landmarks. A common shape is:

```text
X shape = (num_sequences, num_frames, 33, dims)
```

Where `dims` is usually:

- `2`: x, y
- `3`: x, y, z
- `4`: x, y, z, confidence or visibility

The naive approach is to flatten everything:

```text
(T, 33, D) -> (T * 33 * D)
```

and pass that directly to a classifier. This can work for a quick experiment, but it has problems:

- The input dimension grows with sequence length.
- It assumes every frame position means the same thing across videos.
- It is sensitive to camera framing, body size, translation, and scale.
- It does not naturally handle variable-length sequences.
- It encourages overfitting when you have many coordinates but few independent sequences.

A better MLP approach is:

```text
33 keypoints per frame
    -> normalize pose coordinates
    -> shared frame-level MLP encoder
    -> per-frame latent vectors
    -> temporal pooling
    -> sequence-level latent vector
    -> classifier
```

The key phrase is "shared frame-level MLP." The same MLP is applied to every frame. This keeps the model smaller and lets it learn a pose representation that is reused across time.

## Recommended Architecture

Use this architecture first:

```text
Input:
  X: (batch, frames, 33, dims)

Preprocessing:
  center pose around pelvis or mid-hip
  scale by torso length or shoulder-hip distance
  optionally append velocities
  mask or impute low-confidence landmarks

Frame encoder:
  flatten each frame: 33 * dims
  Linear(input_dim, 128)
  BatchNorm or LayerNorm
  ReLU
  Dropout(0.2)
  Linear(128, 64)
  ReLU

Temporal aggregation:
  mean over frames
  std over frames
  optional max over frames

Sequence latent:
  concat(mean_latent, std_latent) -> 128 dimensions if frame latent is 64

Classifier head:
  Linear(128, 64)
  ReLU
  Dropout(0.3)
  Linear(64, num_classes)
```

This is not a transformer, LSTM, or temporal convolution. It is intentionally simple. It gives you a clean answer to the first question: "Does learning a pose latent space help at all?"

## Step 1: Create Honest Baselines

Before adding the MLP encoder, evaluate:

1. Majority-class baseline.
2. Logistic regression on the 82 engineered features.
3. Random Forest or XGBoost on the 82 engineered features.
4. SVM on normalized engineered features.
5. Optional: an MLP on engineered features.

Only keep the keypoint MLP encoder if it beats these baselines on the same split.

Use grouped splits whenever possible:

```text
Good split:
  train videos/subjects and test videos/subjects are disjoint

Bad split:
  frames or clips from the same video appear in both train and test
```

For gait classification, leakage can make a model look much better than it really is.

## Step 2: Normalize Keypoints

Raw pixel coordinates are not ideal for a neural network. The model may learn camera position, crop size, or subject height instead of gait.

For each frame:

1. Compute a body center. For BlazePose 33, a practical center is the midpoint between left hip and right hip.
2. Subtract the center from every landmark.
3. Compute a scale value. Use shoulder width, hip width, torso length, or a bounding-box height.
4. Divide coordinates by the scale.
5. Keep confidence or visibility as an extra input channel.
6. Fill missing or low-confidence points with zeros after centering, and keep their confidence low.

Example:

```python
import numpy as np

LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def normalize_keypoint_sequence(sequence, eps=1e-6):
    """Normalize one sequence shaped (T, 33, D).

    Expected channels:
    - D >= 2: x, y
    - D >= 3: optional z
    - D >= 4: optional confidence or visibility
    """
    x = np.asarray(sequence, dtype=np.float32).copy()

    hips = (x[:, LEFT_HIP, :2] + x[:, RIGHT_HIP, :2]) / 2.0
    shoulders = (x[:, LEFT_SHOULDER, :2] + x[:, RIGHT_SHOULDER, :2]) / 2.0
    torso = np.linalg.norm(shoulders - hips, axis=1, keepdims=True)
    torso = np.maximum(torso, eps)

    x[:, :, :2] = (x[:, :, :2] - hips[:, None, :]) / torso[:, None, :]

    if x.shape[-1] >= 3:
        # z is already relative in many pose models, but scaling keeps magnitude stable.
        x[:, :, 2] = x[:, :, 2] / torso[:, 0, None]

    return x
```

## Step 3: Decide What Goes Into Each Frame

Start simple:

```text
frame_features = [x, y, z, confidence] for each of 33 landmarks
input_dim = 33 * 4 = 132
```

If you only have x/y:

```text
input_dim = 33 * 2 = 66
```

If gait timing matters, add velocity:

```text
position: (x, y, z, confidence)
velocity: delta x, delta y, delta z
input_dim = 33 * 7 = 231
```

Do not start with every possible derived feature. First prove that positions plus confidence work. Then add velocities as a second experiment.

Velocity example:

```python
def append_velocity(sequence):
    """Append per-landmark velocity to a normalized sequence."""
    x = np.asarray(sequence, dtype=np.float32)
    velocity = np.zeros_like(x[:, :, :3])
    velocity[1:] = x[1:, :, :3] - x[:-1, :, :3]
    return np.concatenate([x, velocity], axis=-1)
```

## Step 4: Build the MLP Encoder

The encoder maps one frame into a latent vector:

```text
frame vector -> latent vector
```

For example:

```text
132 -> 128 -> 64
```

Then the sequence model pools all frame latents:

```text
(T, 64) -> mean(64) + std(64) -> 128
```

This gives one compact representation for the whole sequence.

Here is a PyTorch reference implementation:

```python
import torch
from torch import nn


class FrameMLPEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch, frames, input_dim)
        batch, frames, features = x.shape
        x = x.reshape(batch * frames, features)
        z = self.net(x)
        return z.reshape(batch, frames, -1)


class KeypointSequenceMLP(nn.Module):
    def __init__(self, input_dim, num_classes, latent_dim=64, dropout=0.3):
        super().__init__()
        self.encoder = FrameMLPEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            dropout=0.2,
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, mask=None, return_latent=False):
        # x: (batch, frames, input_dim)
        frame_latents = self.encoder(x)

        if mask is None:
            mean_latent = frame_latents.mean(dim=1)
            std_latent = frame_latents.std(dim=1)
        else:
            # mask: (batch, frames), 1 for valid frames and 0 for padded frames
            weights = mask.float().unsqueeze(-1)
            denom = weights.sum(dim=1).clamp_min(1.0)
            mean_latent = (frame_latents * weights).sum(dim=1) / denom
            variance = ((frame_latents - mean_latent[:, None, :]) ** 2 * weights).sum(dim=1) / denom
            std_latent = torch.sqrt(variance.clamp_min(1e-6))

        sequence_latent = torch.cat([mean_latent, std_latent], dim=-1)
        logits = self.classifier(sequence_latent)

        if return_latent:
            return logits, sequence_latent
        return logits
```

This example assumes PyTorch is available. In this project, PyTorch is listed under the optional `pose-enhanced` dependency group. Scikit-learn's `MLPClassifier` is useful as a baseline classifier, but PyTorch is the cleaner choice when you want to explicitly extract and reuse the hidden latent vector.

## Step 5: Prepare Variable-Length Sequences

If every sequence has the same number of frames, you can stack them directly. If not, pad them and keep a mask.

```python
import torch


def collate_keypoint_sequences(batch):
    """Batch items shaped as (sequence, label)."""
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    input_dim = sequences[0].shape[-1]

    x = torch.zeros(len(sequences), max_len, input_dim, dtype=torch.float32)
    mask = torch.zeros(len(sequences), max_len, dtype=torch.bool)

    for i, seq in enumerate(sequences):
        seq = torch.as_tensor(seq, dtype=torch.float32)
        x[i, : len(seq)] = seq
        mask[i, : len(seq)] = True

    y = torch.as_tensor(labels, dtype=torch.long)
    return x, y, mask
```

Before batching, flatten each normalized frame:

```python
def flatten_sequence(sequence):
    """Convert (T, 33, D) to (T, 33 * D)."""
    sequence = np.asarray(sequence, dtype=np.float32)
    return sequence.reshape(sequence.shape[0], -1)
```

## Step 6: Train End-to-End First

The best first experiment is supervised end-to-end training:

```text
normalized keypoints -> MLP encoder -> pooled latent -> classifier head -> label
```

Use cross-entropy loss.

```python
from torch.utils.data import DataLoader


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        logits = model(x, mask=mask)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        logits = model(x, mask=mask)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += len(y)

    return correct / max(total, 1)
```

Recommended optimizer settings:

```python
model = KeypointSequenceMLP(
    input_dim=132,      # 33 landmarks * 4 channels
    num_classes=5,     # update for your labels
    latent_dim=64,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)
```

Use early stopping:

```text
Stop if validation macro-F1 or validation loss does not improve for 10-20 epochs.
```

Accuracy alone can be misleading if classes are imbalanced. Track:

- macro-F1
- balanced accuracy
- confusion matrix
- per-class precision and recall
- calibration if you use probabilities

## Step 7: Export the Latent Space

Once the model is trained, you can extract the sequence-level latent vector and pass it to another classifier.

This is useful for comparing:

```text
engineered 82 features -> RF/SVM/XGBoost
learned MLP latent -> RF/SVM/XGBoost
engineered 82 features + learned latent -> RF/SVM/XGBoost
```

Example:

```python
@torch.no_grad()
def extract_latents(model, loader, device):
    model.eval()
    latents = []
    labels = []

    for x, y, mask in loader:
        x = x.to(device)
        mask = mask.to(device)
        _, z = model(x, mask=mask, return_latent=True)
        latents.append(z.cpu().numpy())
        labels.append(y.numpy())

    return np.concatenate(latents), np.concatenate(labels)
```

Then train a scikit-learn classifier:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


latent_classifier = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced"),
)

latent_classifier.fit(z_train, y_train)
pred = latent_classifier.predict(z_test)
print(classification_report(y_test, pred))
```

## Step 8: Compare Against the 82-Feature Pipeline

Run a table like this:

| Input | Classifier | Split | Macro-F1 | Balanced Accuracy | Notes |
| --- | --- | --- | --- | --- | --- |
| 82 engineered features | Logistic | grouped CV | TBD | TBD | Interpretable baseline |
| 82 engineered features | RF/XGBoost | grouped CV | TBD | TBD | Strong tabular baseline |
| Raw normalized keypoints | MLP encoder + head | grouped CV | TBD | TBD | Learned sequence latent |
| MLP latent | SVM/RF | grouped CV | TBD | TBD | Two-stage learned features |
| 82 features + MLP latent | SVM/RF/XGBoost | grouped CV | TBD | TBD | Hybrid feature set |

The MLP latent approach is a good idea only if it improves the held-out metrics without producing unstable results across random seeds.

Run at least 5 seeds:

```text
seed = 0, 1, 2, 3, 4
report mean and standard deviation
```

If the MLP wins on one seed but loses on others, it is not a reliable improvement yet.

## Step 9: Tune the Architecture Conservatively

Start with the smallest useful network:

```text
input_dim -> 128 -> 64
pool mean + std
128 -> 64 -> num_classes
```

Then tune one axis at a time.

| Parameter | Starting Value | Try Next |
| --- | --- | --- |
| latent_dim | 64 | 32, 128 |
| frame hidden size | 128 | 64, 256 |
| dropout | 0.2-0.3 | 0.1, 0.5 |
| weight_decay | 1e-4 | 1e-5, 1e-3 |
| learning rate | 1e-3 | 3e-4, 3e-3 |
| pooling | mean + std | mean + std + max |
| input channels | x,y,z,confidence | add velocity |

Avoid large networks early:

```text
input_dim -> 512 -> 256 -> 128 -> 64
```

That kind of model may memorize camera/video artifacts unless you have a lot of independent sequences.

## Step 10: Know When Not To Use This

An MLP latent encoder is probably not a good idea if:

- The dataset has very few labeled videos.
- The test accuracy is much lower than training accuracy.
- The model performs well only on random frame-level splits.
- The labels are broad or inconsistent.
- Pose detection quality differs strongly by class.
- You need a clinically explainable decision.
- Engineered features already match or beat the latent model.

In those cases, prefer:

- 82 engineered features plus RF/XGBoost/SVM.
- More careful feature normalization.
- Better train/test grouping.
- Better pose quality checks.
- More labeled sequences.

## Recommended Experiment Order

Use this sequence:

1. Train your current classifiers on the 82 engineered features.
2. Make the split grouped by video, subject, or source when possible.
3. Build the normalized keypoint tensor.
4. Train the small MLP encoder with mean + std pooling.
5. Extract latents from the trained model.
6. Train SVM/RF/XGBoost on those latents.
7. Train SVM/RF/XGBoost on `82 features + latent`.
8. Compare across 5 random seeds.
9. Keep the MLP only if it improves grouped validation and confusion matrix quality.

## Final Practical Recommendation

For AlexPose-style gait classification, I would treat an MLP latent encoder as an experimental feature extractor, not the default classifier.

The best first architecture is:

```text
normalized BLAZEPOSE_33 sequence
  -> per-frame shared MLP: 132 -> 128 -> 64
  -> temporal pooling: mean + std
  -> sequence latent: 128
  -> classifier head or external SVM/RF/XGBoost
```

This gives you a compact learned representation while keeping the model small enough to evaluate honestly. If it beats the 82-feature pipeline on grouped validation, keep exploring. If it does not, the engineered gait features are probably carrying the useful signal more reliably.
