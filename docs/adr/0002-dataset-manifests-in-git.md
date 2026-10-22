# ADR 0002: Dataset versions are manifests committed to git

Status: accepted

## Context
BDD100K images are several GB and live on Kaggle. We still need to know exactly which
images every model was trained and evaluated on, and to notice if that set changes.

## Decision
- A manifest is a CSV per split (image name plus time of day, weather, scene and box
  counts) and a manifest.json with counts and SHA-256 hashes.
- Manifests are built by `pgate data build-manifest` and committed under `manifests/`.
- CI runs `pgate data verify-manifest` on every manifest, so a hand edit fails the build.
- Training refuses to start from a manifest that does not verify.
- The val manifest is the frozen evaluation set. Changing it means a new manifest name
  (bdd100k-v2), never an edit, because results across different eval sets aren't comparable.

## Consequences
Data lineage costs a few hundred KB in git and needs no extra service. Images themselves
are not versioned; we rely on the BDD100K release being immutable.
