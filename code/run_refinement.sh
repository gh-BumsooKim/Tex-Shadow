INPUT="transparent"
CONFIG="config/image_sai.yaml"

ID="<sample_id>"  # replace with your Stage 1 output ID

BASE_INPUT="$ID/save/it1200-export"

MESH="$BASE_INPUT/model.obj"
SAVE_PATH="refined"
OUTDIR="$BASE_INPUT/$SAVE_PATH"

CUDA_VISIBLE_DEVICES=2, python refinement.py --config "$CONFIG" save_path="$SAVE_PATH" outdir="$OUTDIR" input="$INPUT" mesh="$MESH"