N_JOBS=40

# GF7
echo "Generating border basis dataset for GF7 n=3..."
python -m scripts.dataset.generate_dataset \
        --config config/problems/border_basis_GF7_n=3.yaml \
        --task border_basis \
        --save_dir ./data/border_basis/ \
        --n_jobs $N_JOBS \

echo "Generating expansion dataset for GF7 n=3..."
python -m scripts.dataset.generate_dataset \
        --config config/problems/border_basis_GF7_n=3.yaml \
        --task expansion \
        --save_dir ./data/expansion \
        --n_jobs $N_JOBS \

echo "Generating border basis dataset for GF7 n=4..."
python -m scripts.dataset.generate_dataset \
        --config config/problems/border_basis_GF7_n=4.yaml \
        --task border_basis \
        --save_dir ./data/border_basis/ \
        --n_jobs $N_JOBS \

echo "Generating expansion dataset for GF7 n=4..."
python -m scripts.dataset.generate_dataset \
        --config config/problems/border_basis_GF7_n=4.yaml \
        --task expansion \
        --save_dir ./data/expansion \
        --n_jobs $N_JOBS \

echo "Generating border basis dataset for GF7 n=5..."
python -m scripts.dataset.generate_dataset \
        --config config/problems/border_basis_GF7_n=5.yaml \
        --task border_basis \
        --save_dir ./data/border_basis/ \
        --n_jobs $N_JOBS \

echo "Generating expansion dataset for GF7 n=5..."
python -m scripts.dataset.generate_dataset \
        --config config/problems/border_basis_GF7_n=5.yaml \
        --task expansion \
        --save_dir ./data/expansion \
        --n_jobs $N_JOBS \


# GF31
echo "Generating border basis dataset for GF31 n=3..."
python -m scripts.dataset.generate_dataset \
        --config config/problems/border_basis_GF31_n=3.yaml \
        --task border_basis \
        --save_dir ./data/border_basis/ \
        --n_jobs $N_JOBS \

echo "Generating expansion dataset for GF31 n=3..."
python -m scripts.dataset.generate_dataset \
        --config config/problems/border_basis_GF31_n=3.yaml \
        --task expansion \
        --save_dir ./data/expansion \
        --n_jobs $N_JOBS \

echo "Generating border basis dataset for GF31 n=4..."
python -m scripts.dataset.generate_dataset \
        --config config/problems/border_basis_GF31_n=4.yaml \
        --task border_basis \
        --save_dir ./data/border_basis/ \
        --n_jobs $N_JOBS \

echo "Generating expansion dataset for GF31 n=4..."
python -m scripts.dataset.generate_dataset \
        --config config/problems/border_basis_GF31_n=4.yaml \
        --task expansion \
        --save_dir ./data/expansion \
        --n_jobs $N_JOBS \

echo "Generating border basis dataset for GF31 n=5..."
python -m scripts.dataset.generate_dataset \
        --config config/problems/border_basis_GF31_n=5.yaml \
        --task border_basis \
        --save_dir ./data/border_basis/ \
        --n_jobs $N_JOBS \

echo "Generating expansion dataset for GF31 n=5..."
python -m scripts.dataset.generate_dataset \
        --config config/problems/border_basis_GF31_n=5.yaml \
        --task expansion \
        --save_dir ./data/expansion \
        --n_jobs $N_JOBS \


# GF127
echo "Generating border basis dataset for GF127 n=3..."
python -m scripts.dataset.generate_dataset \
        --config config/problems/border_basis_GF127_n=3.yaml \
        --task border_basis \
        --save_dir ./data/border_basis/ \
        --n_jobs $N_JOBS \

echo "Generating expansion dataset for GF127 n=3..."
python -m scripts.dataset.generate_dataset \
        --config config/problems/border_basis_GF127_n=3.yaml \
        --task expansion \
        --save_dir ./data/expansion \
        --n_jobs $N_JOBS \

echo "Generating border basis dataset for GF127 n=4..."
python -m scripts.dataset.generate_dataset \
        --config config/problems/border_basis_GF127_n=4.yaml \
        --task border_basis \
        --save_dir ./data/border_basis/ \
        --n_jobs $N_JOBS \

echo "Generating expansion dataset for GF127 n=4..."
python -m scripts.dataset.generate_dataset \
        --config config/problems/border_basis_GF127_n=4.yaml \
        --task expansion \
        --save_dir ./data/expansion \
        --n_jobs $N_JOBS \

echo "Generating border basis dataset for GF127 n=5..."
python -m scripts.dataset.generate_dataset \
        --config config/problems/border_basis_GF127_n=5.yaml \
        --task border_basis \
        --save_dir ./data/border_basis/ \
        --n_jobs $N_JOBS \

echo "Generating expansion dataset for GF127 n=5..."
python -m scripts.dataset.generate_dataset \
        --config config/problems/border_basis_GF127_n=5.yaml \
        --task expansion \
        --save_dir ./data/expansion \
        --n_jobs $N_JOBS \


