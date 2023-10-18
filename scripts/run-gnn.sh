INPUT_FILE=/eos/user/d/dapullia/tp_dataset/snana_hits.txt
OUTPUT_FOLDER=/eos/user/d/dapullia/gnn_approach/snana_hits/
MODEL_NAME=model_unbalanced
LOAD_MODEL=false
BALANCE_TRAINING_SET=true   
N_TPS_TO_READ=5000000
N_EPOCHS_TO_TRAIN=400
N_EPOCHS_TO_TEST=1
DATASET_FOLDER=/eos/user/d/dapullia/gnn_approach/snana_hits/data/
BATCH_SIZE=32
TRAIN_TEST_SPLIT=0.8
EARLY_STOPPING_PATIENCE=8
# Function to print help message

print_help() {
    echo "*****************************************************************************"
    echo "Usage: run-gnn.sh -i <input_file> -o <output_folder> [-h]"
    echo "  -i  input file"
    echo "  -o  output folder"
    echo "  -h  print this help message"
    echo "*****************************************************************************"
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input_file)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--OUTPUT_FOLDER)
            OUTPUT_FOLDER="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            ;;
        *)
            shift
            ;;
    esac
done

# stop execution if fundamental variables are not set
if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FOLDER" ]
then
    echo "Usage: ./run-classifier.sh -i <input_file> -o <output_folder> [-h]"
    exit 0
fi

# if load_model is true then add --load_model
if [ "$LOAD_MODEL" = true ] ; then
    LOAD_MODEL_FLAG="--load_model"
else
    LOAD_MODEL_FLAG=""
fi

# if balance_training_set is true then add --balance_training_set

if [ "$BALANCE_TRAINING_SET" = true ] ; then
    BALANCE_TRAINING_SET_FLAG="--balance_training_set"
else
    BALANCE_TRAINING_SET_FLAG=""
fi

# move to the folder, run and come back to scripts
cd ../python/
python gnn_classifier.py --input_data $INPUT_FILE --output_folder $OUTPUT_FOLDER --dataset_folder $DATASET_FOLDER --model_name $MODEL_NAME --n_tps_to_read $N_TPS_TO_READ --n_epochs_to_train $N_EPOCHS_TO_TRAIN --n_epochs_to_test $N_EPOCHS_TO_TEST $LOAD_MODEL_FLAG $BALANCE_TRAINING_SET_FLAG --batch_size $BATCH_SIZE --train_test_split $TRAIN_TEST_SPLIT --early_stopping_patience $EARLY_STOPPING_PATIENCE
cd ../scripts



