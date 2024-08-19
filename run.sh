################################ Best AUC and F1 for CAPF ###############################
############# MIMIC-III dataset ######################
python train.py --model Dip_g --num_path 3 --input_dim 2850 --hidden_dim 256 --output_dim 90 --batch_size 4 --lambda 0.5 --K 2 --data_type mimic-iii --dropout_ratio 0.1 --decay 0.0001
############# MIMIC-IV dataset ######################
python train.py --model Dip_g --num_path 4 --input_dim 1992 --hidden_dim 256 --output_dim 80 --batch_size 2 --lambda 0.5 --K 3 --data_type mimic-iv --dropout_ratio 0.1 --decay 0.0001

################################ Best AUC and F1 for Baseline Methods #################################
############# Lstm ###############
python train.py --model LSTM --input_dim 2850 --hidden_dim 256 --output_dim 90 --batch_size 8 --data_type mimic-iii --only_dipole --decay 0.0001
python train.py --model LSTM --input_dim 1992 --hidden_dim 256 --output_dim 80 --batch_size 8 --data_type mimic-iv --only_dipole --decay 0.0001
############# Dipole ###############
python train.py --model Dip_g --input_dim 2850 --hidden_dim 256 --output_dim 90 --batch_size 8 --data_type mimic-iii --only_dipole --decay 0.0001
python train.py --model Dip_g --input_dim 1992 --hidden_dim 256 --output_dim 80 --batch_size 8 --data_type mimic-iv --only_dipole --decay 0.0001
############# Retain ###############
python train.py --model Retain --input_dim 2850 --hidden_dim 256 --output_dim 90 --batch_size 8 --data_type mimic-iii --only_dipole --decay 0.0001
python train.py --model Retain --input_dim 1992 --hidden_dim 256 --output_dim 80 --batch_size 8 --data_type mimic-iv --only_dipole --decay 0.0001
############# GraphCare ###############
python train.py --model GraphCare --input_dim 2850 --hidden_dim 256 --output_dim 90 --batch_size 8 --data_type mimic-iii --dropout_ratio 0.3 --gamma_GraphCare 0.1 --decay 0.0001
python train.py --model GraphCare --input_dim 1992 --hidden_dim 256 --output_dim 80 --batch_size 8 --data_type mimic-iv --dropout_ratio 0.3 --gamma_GraphCare 0.1 --decay 0.0001
############# MedPath ###############
python train.py --model MedPath --input_dim 2850 --hidden_dim 256 --output_dim 90 --K 2 --batch_size 8 --data_type mimic-iii --dropout_ratio 0.3 --decay 0.0001
python train.py --model MedPath --input_dim 1992 --hidden_dim 256 --output_dim 80 --K 3 --batch_size 8 --data_type mimic-iv --dropout_ratio 0.3 --decay 0.0001
############# HAR ###############
python train.py --model StageAware --input_dim 2850 --hidden_dim 256 --output_dim 90 --batch_size 8 --data_type mimic-iii --lambda_HAR 0.1 --decay 0.0001
python train.py --model StageAware --input_dim 1992 --hidden_dim 256 --output_dim 80 --batch_size 8 --data_type mimic-iv --lambda_HAR 0.1 --decay 0.0001

############################### Interpretation ####################################
############# MIMIC-III dataset ######################
python train.py --model Dip_g --num_path 3 --input_dim 2850 --hidden_dim 256 --output_dim 90 --batch_size 4 --lambda1 0.5 --lambda2 0.5 --K 3 --data_type mimic-iii --dropout_ratio 0.1 --alpha_CAPF 0.2  --show_interpretation --decay 0.0001
############# MIMIC-IV dataset ######################
python train.py --model Dip_g --num_path 4 --input_dim 1992 --hidden_dim 256 --output_dim 80 --batch_size 2 --lambda1 0.5 --lambda2 0.5 --K 3 --data_type mimic-iv --dropout_ratio 0.1 --alpha_CAPF 0.2 --show_interpretation --decay 0.0001