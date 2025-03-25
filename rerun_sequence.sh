for config in "$@"
do
    echo "RERUN $config"
    python main.py --config=$config --ignore_results=True
done

# CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" bash ./train_sequence.sh config/ntu60/xsub/joint/vagcn/train.yaml