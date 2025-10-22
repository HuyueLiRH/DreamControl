conda activate omnicontrol
python -m sample.generate --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt --num_repetitions 1 --task kick --output_dir sample/kick_final --batch_size 128
conda activate env_isaaclab
cd sample/kick_final1
rm -r *.pkl
cd ../kick_final2
rm -r *.pkl
cd ../kick_final
python3 retarget.py
cd ../kick_final1
python3 refine_motions.py
cd ..
