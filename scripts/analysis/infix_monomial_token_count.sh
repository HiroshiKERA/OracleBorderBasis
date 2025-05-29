python -m src.misc.monomial_token_counter \ 
    --config_path data/prod/GF7_n=2/config.yaml \
    --data_path data/prod/GF7_n=2/test \
    --output_dir data/prod/GF7_n=2
    
python -m src.misc.monomial_token_counter \ 
    --config_path data/prod/GF7_n=3/config.yaml \
    --data_path data/prod/GF7_n=3/test \
    --output_dir data/prod/GF7_n=3

python -m src.misc.monomial_token_counter \ 
    --config_path data/prod/GF7_n=4/config.yaml \
    --data_path data/prod/GF7_n=4/test \
    --output_dir data/prod/GF7_n=4

python -m src.misc.monomial_token_counter \ 
    --config_path data/prod/GF7_n=5/config.yaml \
    --data_path data/prod/GF7_n=5/test \
    --output_dir data/prod/GF7_n=5
