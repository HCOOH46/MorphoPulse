#!/bin/bash
MODEL="${MODEL:-foundation_model.pt}"
TIME="${TIME:-default_run}"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets/downstream}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/artifacts/checkpoints/foundation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/outputs/foundation}"

if [[ "$MODEL" == *.pt ]]; then
    MODEL_FILE="$MODEL"
else
    MODEL_FILE="${MODEL}.pt"
fi

for DATASET in wesad dalia sdb ppg-bp mimic_af lexin_af vital; do
    for split in test val train; do
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python feature_extraction_pm.py "$CHECKPOINT_ROOT/$TIME/$MODEL_FILE" 0 "$DATASET" "$split" "$OUTPUT_ROOT/features/$DATASET" None None True True 125 125 False 0
    done
    if [ "$DATASET" = "sdb" ]; then
        python fea_combine.py -l patient -f "$OUTPUT_ROOT/features/$DATASET/${MODEL_FILE%.pt}"
        continue
    elif [ "$DATASET" = "ppg-bp" ]; then
        python fea_combine.py -l patient -f "$OUTPUT_ROOT/features/$DATASET/${MODEL_FILE%.pt}"
        continue
    elif [ "$DATASET" = "vital" ]; then
        python fea_combine.py -l patient -f "$OUTPUT_ROOT/features/$DATASET/${MODEL_FILE%.pt}"
        continue
    fi
    python fea_combine.py -l segment -f "$OUTPUT_ROOT/features/$DATASET/${MODEL_FILE%.pt}"
done

python outcome_classification_all.py "${MODEL_FILE%.pt}" binary
