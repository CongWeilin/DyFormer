GPU_ID=0
################################################
################################################
################################################
PYTHON_FILE=directly_train.py

DATASET=UCI_13
for TIME_STEP in {3..13}; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_FILE --dataset $DATASET --time_step $TIME_STEP
done

DATASET=Yelp_16
for TIME_STEP in {4..16}; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_FILE --dataset $DATASET --time_step $TIME_STEP
done


DATASET=ML_10M_13
for TIME_STEP in {4..13}; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_FILE --dataset $DATASET --time_step $TIME_STEP
done

################################################
################################################
################################################
PYTHON_FILE=pretrain_finetune.py

DATASET=UCI_13
for TIME_STEP in {3..13}; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_FILE --dataset $DATASET --time_step $TIME_STEP
done

DATASET=Yelp_16
for TIME_STEP in {4..16}; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_FILE --dataset $DATASET --time_step $TIME_STEP
done


DATASET=ML_10M_13
for TIME_STEP in {4..13}; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_FILE --dataset $DATASET --time_step $TIME_STEP
done