GPU_ID=0
################################################
################################################
################################################
PYTHON_FILE=run_single_step_baselines.py

MODEL_NAME=EvolveGCN_O # DySAT

DATASET=UCI_13
for TIME_STEP in {3..13}; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_FILE --dataset $DATASET --time_step $TIME_STEP --model_name $MODEL_NAME
done

DATASET=Yelp_16
for TIME_STEP in {4..16}; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_FILE --dataset $DATASET --time_step $TIME_STEP --model_name $MODEL_NAME
done


DATASET=ML_10M_13
for TIME_STEP in {4..13}; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_FILE --dataset $DATASET --time_step $TIME_STEP --model_name $MODEL_NAME
done