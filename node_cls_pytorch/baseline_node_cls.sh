

GPU_ID=0

DATASET=reddit_classification
# MODLE_NAME=EvolveGCN_O

# DATASET=wiki_classification


# MODLE_NAME=DySAT
# PYTHON_FILE=train_models_baseline_node_cls.py

# for SEED in 123 231 321
# do
#     for TIME_STEP in {5..11}; do
#         CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_FILE --dataset $DATASET --time_step $TIME_STEP --model_name $MODLE_NAME --seed $SEED
#     done
# done

# MODLE_NAME=EvolveGCN_O
# PYTHON_FILE=train_models_baseline_node_cls.py

# for SEED in 123 231 321
# do
#     for TIME_STEP in {5..11}; do
#         CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_FILE --dataset $DATASET --time_step $TIME_STEP --model_name $MODLE_NAME --seed $SEED
#     done
# done



#####################################

# PYTHON_FILE=pretrain_models_graph_trans.py

# for SEED in 123
# do
#     for TIME_STEP in {4..10}; do
#         CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_FILE --dataset $DATASET --time_step $TIME_STEP --seed $SEED 
#     done
# done

PYTHON_FILE=train_models_graph_trans_node_cls.py

for SEED in 123 231 321
do
    for TIME_STEP in {5..11}; do
        CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_FILE --dataset $DATASET --time_step $TIME_STEP --seed $SEED --use_pretrain 0
    done
done


# PYTHON_FILE=train_models_graph_trans_node_cls.py

# for SEED in 123 231 321
# do
#     for TIME_STEP in {5..11}; do
#         CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_FILE --dataset $DATASET --time_step $TIME_STEP --seed $SEED --use_pretrain 1
#     done
# done