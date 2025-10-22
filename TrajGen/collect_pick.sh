conda activate omnicontrol
python -m sample.generate --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt --num_repetitions 1 --task pick --output_dir sample/pick_final --batch_size 128
conda activate env_isaaclab
cd sample/pick_final1
rm -r *.pkl
cd ../pick_final2
rm -r *.pkl
cd ../pick_final
python3 retarget.py
cd ../pick_final1
python3 refine_motions.py
cd ..
