# conda activate omnicontrol
python -m sample.generate --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt --num_repetitions 1 --task punch --output_dir sample/punch_final --batch_size 128
conda activate env_isaaclab
cd sample/punch_final1
rm -r *.pkl
cd ../punch_final2
rm -r *.pkl
cd ../punch_final
python3 retarget.py
cd ../punch_final1
python3 refine_motions.py
cd ..
