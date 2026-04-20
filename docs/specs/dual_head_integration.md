# Spec: ACES Transformer & Dual-Head Integration

## 1. Overview
The ACES Transformer is the second major component of the LuminaScale pipeline. While the Dequantization-Net (Head 1) focuses on structural restoration and bit-depth expansion, the ACES Transformer (Head 2) performs "blind" color space normalization to the ACES (Academy Color Encoding System) workspace.

## 2. Integration Architecture
To maximize efficiency and flexibility, the repository will support three distinct training phases.

### Phase 1: Task-Specific Pre-training (Decoupled)
*   **Goal**: Independent convergence of both heads to baseline performance.
*   **Execution**: Two separate training scripts and trainers sharing the same WebDataset pipeline.
    *   `scripts/train_dequantization_net.py` -> `DequantizationTrainer`
    *   `scripts/train_aces_transformer.py` -> `AcesTransformerTrainer`
*   **Benefit**: Computational simplicity and isolation of "blind" color loss experimentation.

### Phase 2: Joint Fine-tuning (Composition)
*   **Goal**: Optimize the interaction between bit-depth expansion and color normalization.
*   **Execution**: A unified trainer (`JointLuminaScaleTrainer`) that loads both pre-trained models.
*   **Data Flow**:
    1.  `Input (8-bit LQ)` -> `Dequantization-Net` -> `Restored 32-bit (Linear-ish)`
    2.  `Restored 32-bit` -> `ACES Transformer` -> `Final ACES AP1 output`
*   **Optimization**: Manual optimization in PyTorch Lightning to handle separate backpropagation steps or alternating updates for the two different loss objectives.

## 3. Training Logic & Code Structure
The `src/luminascale/training/` directory will be refactored to support this progression:

1.  **`BaseLuminaScaleTrainer`**: Shared logic for dataset loading (WebDataset), logging, and throughput monitoring.
2.  **`DequantizationTrainer`**: Inherits from `BaseLuminaScaleTrainer`. Focused on reconstruction loss ($L1$, Charbonnier).
3.  **`AcesTransformerTrainer`**: Inherits from `BaseLuminaScaleTrainer`. Focused on color statistics and "blind" perceptual losses.
4.  **`JointLuminaScaleTrainer`**: Composes both models. Handles the pipeline flow where the output of Head 1 is the input to Head 2.

## 4. Hardware & Scaling Strategy
*   **Compute**: Leverage multi-GPU HPC nodes.
*   **Bottleneck Mitigation**: Since data loading is the primary bottleneck, separate training runs will use high `num_workers` counts. Joint fine-tuning will maximize GPU utilization by filling the pipeline with both models simultaneously.

## 5. Metadata & Data Requirements
Both heads will utilize the existing WebDataset (.tar shards) containing high-dynamic-range EXR files.
*   **Head 1 Target**: Original High-Bit EXR (Structure).
*   **Head 2 Target**: "Blind" normalization (Statistics/Perceptual).

## 6. Configuration Management (Hydra)
To manage multiple tasks, we will use a hierarchical Hydra structure.

### Proposed Config Tree:
```text
configs/
├── task/
│   ├── dequant.yaml       # Phase 1: BDE specific params
│   ├── aces.yaml          # Phase 1: ACES specific params
│   └── joint.yaml         # Phase 2: Joint weights/LRs
├── model/
│   ├── dequant_net.py     # Arch params for Head 1
│   └── aces_trans.py      # Arch params for Head 2
└── train_cfg.yaml         # Master config (Composition)
```

*   **Task Switching**: Using `python scripts/train.py task=joint` will automatically compose the required heads.
*   **Checkpoint Overrides**: The `joint.yaml` will contain paths to pre-trained checkpionts for initialization (`dequant_pretrained_path`, `aces_pretrained_path`).

## 7. Experiment Tracking (TensorBoard)
To maintain total isolation and simplify debugging, each training task will log to its own dedicated directory structure.

### Isolated Logging Strategy:
*   **Directory Scoping**:
    *   `outputs/training/dequant_runs/` -> (Pre-training Head 1)
    *   `outputs/training/aces_runs/`   -> (Pre-training Head 2)
    *   `outputs/training/joint_runs/`  -> (Joint Fine-tuning Phase)

### Multi-Instance Visualization:
*   **Execution**: Each task creates a completely independent TensorBoard logger instance.
*   **User Interface**: Users can view the tasks individually or compare them by pointing TensorBoard at the root `outputs/training/` folder, which will list all three tasks as distinct "runs" in the sidebar.
*   **Checkpoint Isolation**: Model checkpoints (.ckpt) will be bundled within these same task-specific folders, making it impossible to accidentally overwrite a pre-trained Dequant model with a Joint one.
