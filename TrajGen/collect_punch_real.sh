conda activate omnicontrol
python -m sample.generate --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt --num_repetitions 1 --task punch_real --output_dir sample/punch_real --batch_size 1500
conda activate env_isaaclab
cd sample/punch_real1
# rm -r *.pkl
cd ../punch_real2
rm -r *.pkl
cd ../punch_real
python3 retarget.py
cd ../punch_real1
python3 refine_motions.py
cd ..
