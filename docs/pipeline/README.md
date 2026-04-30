# Pipeline Walkthrough

This document traces the main inference pipeline end to end and maps each stage to the functions that implement it.

The main entrypoint is `python -m vascx_models run DATA_PATH OUTPUT_PATH`, which resolves to:

- `vascx_models/__main__.py`
- `vascx_models/cli.py:run`
- `vascx_models/pipeline.py:run_pipeline`

If you want a runnable code example of the same sequence, see `tests/test_e2e.py:test_single_image_pipeline_smoke`.

## High-Level Flow

1. Bootstrap the runtime environment.
2. Parse CLI flags and load config.
3. Resolve input images from a directory or CSV.
4. Preprocess images into a normalized PNG workspace.
5. Select the inference device.
6. Run image quality estimation.
7. Run vessel and artery-vein segmentation.
8. Run optic disc segmentation.
9. Estimate disc geometry and generate disc-centered circles.
10. Trace vessel paths and compute vessel metrics.
11. Run fovea detection.
12. Render overlays and write all outputs.

## Stage By Stage

### 1. Runtime bootstrap

Functions:

- `vascx_models.runtime.configure_runtime_environment`
- `vascx_models.__main__`

What happens:

- The package creates local cache directories such as `.mplconfig` and `.cache`.
- It creates or refreshes `model_releases/` symlinks so the inference code can load model weights from one flat directory, even though the repository stores them in task-specific folders.
- This runs before the CLI command body, so later stages can assume model discovery is already configured.

### 2. CLI parsing and config loading

Functions:

- `vascx_models.cli.run`
- `vascx_models.config.load_app_config`
- `vascx_models.cli._pipeline_dependencies`

What happens:

- The CLI reads command flags such as `--no-preprocess`, `--device`, and `--config`.
- Config is loaded from an explicit path, the current working directory, or the repository root.
- A dependency bundle is assembled so `run_pipeline` receives concrete implementations for preprocessing, inference, metrics, and overlay generation.

### 3. Output directory and stage gating

Function:

- `vascx_models.pipeline.run_pipeline`

What happens:

- The output directory is created.
- The pipeline decides which subdirectories to create based on the requested stages.
- File paths for outputs such as `bounds.csv`, `quality.csv`, `disc_geometry.csv`, and metric CSVs are computed up front.

### 4. Input discovery

Function:

- `vascx_models.pipeline.run_pipeline`

What happens:

- If `DATA_PATH` is a CSV, the pipeline expects a `path` column and optionally uses an `id` column.
- If `DATA_PATH` is a directory, it scans for supported image extensions and uses each stem as the image ID.
- The rest of the pipeline uses these IDs to keep all per-image outputs aligned.

### 5. Preprocessing

Functions:

- `vascx_models.pipeline.run_pipeline`
- `rtnls_fundusprep.cli._run_preprocessing`

What happens:

- Raw fundus images are normalized into PNGs under `preprocessed_rgb/`.
- Crop or boundary metadata is written to `bounds.csv`.
- If preprocessing is disabled, the original images are used directly for inference.

### 6. Device selection

Functions:

- `vascx_models.models.inference.available_device_types`
- `vascx_models.models.inference.resolve_device`

What happens:

- The pipeline checks whether CUDA and MPS are available.
- `auto` chooses CUDA first, then MPS, then CPU.
- A requested but unavailable accelerator fails explicitly instead of silently falling back.

### 7. Quality estimation

Function:

- `vascx_models.models.inference.run_quality_estimation`

What happens:

- The quality classifier is loaded.
- Images are batched through the model.
- The pipeline writes `quality.csv` with three quality outputs per image.

### 8. Vessel and artery-vein segmentation

Function:

- `vascx_models.models.inference.run_segmentation_vessels_and_av`

What happens:

- Two segmentation models run on the same processed image set.
- One produces a binary vessel mask.
- The other produces artery, vein, and overlap labels.
- Output files are written to `vessels/` and `artery_vein/`.

### 9. Optic disc segmentation

Function:

- `vascx_models.models.inference.run_segmentation_disc`

What happens:

- The disc model predicts the optic disc mask for each image.
- Masks are restored to the image coordinate space and written to `disc/`.

## 10. Disc geometry and disc-centered circles

This is the point where the pipeline stops being just a set of model predictions and starts building the geometric frame used by downstream measurements.

Functions:

- `vascx_models.geometry.disc_circles.estimate_disc_geometry`
- `vascx_models.geometry.disc_circles.generate_disc_circles`

Inputs:

- `disc/<image_id>.png`
- configured overlay circles from `AppConfig.overlay.circles`

Outputs:

- `disc_geometry.csv`
- `disc_circles/<circle_name>/<image_id>.png`

Processing logic:

1. Each binary disc mask is loaded.
2. The disc center is estimated as the mean of all foreground disc pixels.
3. The disc radius is estimated from disc area using an equivalent-circle radius, not from a fitted edge.
4. For each configured circle, the radius is scaled as `disc_radius * circle.diameter`.
5. A one-pixel-thick circle mask is rasterized and saved.
6. The numeric disc center, disc radius, and derived circle radii are stored in `disc_geometry.csv`.

Why this matters:

- Every downstream vessel metric is defined relative to this disc-centered coordinate system.
- The inner and outer metric circles do not come from vessel geometry. They come from disc geometry.
- If the disc mask is empty, later geometric stages intentionally skip that image because there is no stable reference annulus.

## 11. Vessel metric orchestration

This is the main branching point for the geometry-heavy half of the pipeline.

Functions:

- `vascx_models.pipeline.compute_and_save_vessel_metrics`
- `vascx_models.metrics.vessel_widths.measure_vessel_widths_between_disc_circle_pair`
- `vascx_models.metrics.vessel_tortuosities.measure_vessel_tortuosities_between_disc_circle_pair`
- `vascx_models.metrics.vessel_branching.measure_vessel_branching_between_disc_circle_pair`

Inputs:

- `vessels/`
- `artery_vein/`
- `disc_geometry.csv`
- metric-specific inner and outer circle names from config

Outputs:

- `vessel_widths.csv`
- `vessel_widths_summary.csv`
- `vessel_tortuosities.csv`
- `vessel_tortuosity_summary.csv`
- `vessel_branching.csv`
- `vessel_branching_widths.csv`

Control logic:

1. The pipeline resolves the configured inner and outer circle names into actual `OverlayCircle` definitions.
2. It removes stale metric files for any metric families that are disabled.
3. If all vessel metric sections are disabled, the function exits early.
4. If profile-based width measurement is enabled, it also resolves the RGB image source for profile sampling.
5. It then runs width, tortuosity, and branching independently, but all three use the same disc-centered annulus idea.

The important design point is that each metric family uses its own path-tracing rule. They all start from vessel masks and disc geometry, but they do not all trace the same kinds of paths.

## 12. Width measurement logic

Functions:

- `vascx_models.metrics.vessel_widths.measure_vessel_widths_between_disc_circle_pair`
- `vascx_models.geometry.vessel_paths.trace_vessel_paths_between_disc_circle_pair`
- `vascx_models.metrics.vessel_widths.compute_revised_crx_from_widths`

### 12.1 Inputs are split by vessel type

The width stage does not measure the combined vessel mask directly.

Processing logic:

1. Load the binary vessel mask.
2. Load the artery-vein label mask.
3. Split the vessel pixels into artery-only and vein-only masks.
4. Measure arteries and veins separately so the final summary can compute CRAE and CRVE independently.

### 12.2 Path tracing between the two circles

Function:

- `vascx_models.geometry.vessel_paths.trace_vessel_paths_between_disc_circle_pair`

This function defines what counts as a width-measurable vessel connection.

Processing logic in English:

1. Skeletonize the typed vessel mask so each vessel becomes a one-pixel-wide centerline.
2. Compute the distance from every pixel to the disc center.
3. Keep only skeleton pixels whose radius lies inside the chosen annulus between the inner and outer circles.
4. Split that annular skeleton into connected components.
5. For each component, label candidate boundary nodes near the inner circle and near the outer circle.
6. Prune away dead-end pixels that cannot participate in an inner-to-outer connection.
7. Keep only groups that still connect one side of the annulus to the other.
8. Trace trunk-like segments between key nodes.
9. Retain only those segments whose endpoints include the inner side, then orient them from the inner circle outward.

What this means operationally:

- A vessel branch that never spans the annulus is ignored.
- Dead-end side branches are removed before sampling widths.
- If a vessel forks inside the annulus, the retained width path is trunk-focused rather than daughter-focused.

### 12.3 Sampling widths along each retained path

Once a path is retained, the width stage samples interior points along it.

Processing logic:

1. Compute cumulative length along the traced path.
2. Choose interior sample positions based on `samples_per_connection`.
3. Estimate the local tangent direction near each sample point.
4. Convert that tangent into a normal direction.
5. Walk outward in the positive and negative normal directions until the vessel mask boundary is found.
6. Refine each boundary location subpixel-wise.
7. Use the two boundary points to compute the final width.
8. Save not just the width, but also the measurement line endpoints and diagnostic fields.

If `vessel_widths.method` is `profile`, the same traced geometry is used, but the width is estimated from an image-intensity profile rather than only from the mask. The path-tracing rule is still the gatekeeper for what gets measured.

### 12.4 Aggregating widths into CRAE and CRVE

Function:

- `vascx_models.metrics.vessel_widths.compute_revised_crx_from_widths`

Processing logic:

1. Group per-sample width rows by image, vessel type, and connection index.
2. Compute one mean width per retained connection.
3. Rank those mean widths within arteries and within veins.
4. Select up to the largest six vessels for each type.
5. Apply the revised Knudtson-style recursive equivalent formula.
6. Save the per-connection table and the artery or vein equivalent summary.

The summary file is written to `vessel_widths_summary.csv`, even though the metric names inside it are `CRAE` and `CRVE`.

## 13. Tortuosity measurement logic

Functions:

- `vascx_models.metrics.vessel_tortuosities.measure_vessel_tortuosities_between_disc_circle_pair`
- `vascx_models.geometry.vessel_paths.trace_vessel_tortuosity_paths_between_disc_circle_pair`
- `vascx_models.metrics.vessel_tortuosities.summarize_vessel_tortuosities`

The tortuosity stage uses a different tracing rule than widths because it wants meaningful path segments, including valid branches, rather than just trunk segments for width sampling.

### 13.1 Path tracing rule for tortuosity

Function:

- `vascx_models.geometry.vessel_paths.trace_vessel_tortuosity_paths_between_disc_circle_pair`

Processing logic in English:

1. Skeletonize the typed vessel mask.
2. Restrict the skeleton to the configured annulus.
3. Build connected components.
4. Label inner-boundary and outer-boundary nodes.
5. Prune dead ends that cannot support an inner-to-outer traversal.
6. For each remaining group, require a single-inner rooted tree structure.
7. Split that rooted tree into one-to-one key-node segments.
8. Discard ambiguous patterns such as cycles, merges, and many-to-one structures.

Why this differs from width tracing:

- Tortuosity wants ordered path segments that reflect local vessel course.
- A valid bifurcation can produce multiple measurable segments.
- The stage prefers unambiguous topology over maximum coverage.

### 13.2 Computing tortuosity

For each retained path segment:

1. Compute total path length along the skeleton.
2. Compute straight-line chord length from the first point to the last point.
3. Compute tortuosity as `path_length / chord_length`.
4. Save the segment endpoints, path length, chord length, tortuosity, and vessel type.

### 13.3 Summarizing tortuosity

The summary stage groups segments by image and vessel type.

Processing logic:

1. Keep only rows with finite tortuosity and positive path length.
2. Count retained segments.
3. Count unique starting points.
4. Sum retained path length.
5. Compute a length-weighted mean tortuosity.
6. Emit `TORTA` for arteries and `TORTV` for veins.

This summary is written to `vessel_tortuosity_summary.csv`.

## 14. Branching measurement logic

Functions:

- `vascx_models.metrics.vessel_branching.measure_vessel_branching_between_disc_circle_pair`
- `vascx_models.geometry.vessel_paths.trace_vessel_branching_points_between_disc_circle_pair`

The branching stage is stricter than the tortuosity stage because it is looking for explicit bifurcation geometry: one parent branch splitting into two daughters.

### 14.1 Tracing rooted bifurcations

Function:

- `vascx_models.geometry.vessel_paths.trace_vessel_branching_points_between_disc_circle_pair`

Processing logic in English:

1. Skeletonize the typed vessel mask.
2. Restrict to the annulus between the chosen circles.
3. Build connected components.
4. Label inner and outer boundary nodes.
5. Prune unreachable dead ends.
6. Require each measured group to have exactly one inner-side root.
7. Build a rooted parent-child tree from that root.
8. Search key nodes for the specific pattern where one upstream parent continues into exactly two outer-reaching child branches.
9. Trace the parent path back toward the disc.
10. Trace the two daughter paths outward.
11. Discard anything ambiguous, cyclic, merged, or not truly bifurcating.

This yields a clean representation of each retained branching point:

- junction coordinate
- one parent path
- two daughter paths

### 14.2 Measuring widths and angles around a branch point

Once a bifurcation is retained:

1. Sample several width locations along the parent branch and along each daughter branch.
2. Estimate local tangents and normals at those sample positions.
3. Measure widths with the same boundary-search style used for mask widths.
4. Take the median valid width for the parent and for each daughter.
5. Compute the angle between the two daughter branch directions.
6. Compute the branching coefficient as `(daughter_1_width^2 + daughter_2_width^2) / parent_width^2` when the parent width is valid.

The stage writes:

- `vessel_branching.csv` for per-bifurcation records
- `vessel_branching_widths.csv` for the underlying sampled width audit rows

## 15. Fovea detection

Function:

- `vascx_models.models.inference.run_fovea_detection`

What happens:

- A heatmap regression model predicts the fovea location.
- The extracted coordinates are written to `fovea.csv`.

This stage is logically separate from the annulus-based vessel metrics. It contributes to overlays and downstream interpretation, but not to width, tortuosity, or branching calculations.

## 16. Overlay rendering

Functions:

- `vascx_models.pipeline.render_metric_overlays`
- `vascx_models.overlays.utils.batch_create_overlays`
- `vascx_models.overlays.utils.create_fundus_overlay`

This stage turns the numeric outputs back into visual evidence.

### 16.1 Overlay orchestration

The pipeline first decides which metric-specific overlays should be rendered.

Processing logic:

1. Build a base overlay configuration.
2. Restrict circle sets for width, tortuosity, and branching overlays to the circles relevant to each metric.
3. Load optional fovea coordinates.
4. For width overlays, optionally filter down to the vessel measurements selected for CRAE or CRVE.
5. Call the batch renderer once for the main overlay and again for metric-specific overlays when data exists.

### 16.2 Per-image overlay rendering

For each processed RGB image, `create_fundus_overlay` does the following:

1. Load the RGB image.
2. Paint artery and vein masks if present.
3. Paint the optic disc mask if present.
4. Paint each configured disc circle if present.
5. Rasterize vessel width measurement lines and clip them to the vessel mask.
6. Rasterize branching width samples.
7. Draw branching junction markers.
8. Draw the fovea marker.
9. For tortuosity overlays, trace the vessel skeleton between each retained segment endpoint pair and also draw the straight endpoint chord.
10. Draw daughter-angle guide lines for branching overlays.
11. Save the composite image.

Outputs:

- `overlays/`
- `vessel_width_overlays/`
- `vessel_tortuosity_overlays/`
- `vessel_branching_overlays/`

These overlays are the easiest way to inspect whether the geometric rules used in steps 10 through 14 match the expected vessel structures in a given image.

## Final Output Set

The exact output set depends on which stages are enabled, but a full run can produce:

- `preprocessed_rgb/`
- `vessels/`
- `artery_vein/`
- `disc/`
- `disc_circles/`
- `overlays/`
- `vessel_width_overlays/`
- `vessel_tortuosity_overlays/`
- `vessel_branching_overlays/`
- `bounds.csv`
- `quality.csv`
- `disc_geometry.csv`
- `vessel_widths.csv`
- `vessel_widths_summary.csv`
- `vessel_tortuosities.csv`
- `vessel_tortuosity_summary.csv`
- `vessel_branching.csv`
- `vessel_branching_widths.csv`
- `fovea.csv`

## Minimal Call Chain

`vascx_models.__main__`

-> `vascx_models.runtime.configure_runtime_environment`

-> `vascx_models.cli.run`

-> `vascx_models.config.load_app_config`

-> `vascx_models.pipeline.run_pipeline`

-> preprocessing

-> quality estimation

-> vessel and artery-vein segmentation

-> disc segmentation

-> disc geometry and circle generation

-> vessel metric orchestration

-> width, tortuosity, and branching measurement

-> fovea detection

-> overlay generation
