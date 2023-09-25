for config in "$@"
do
    echo "$config"
    python main.py --config=$config
done