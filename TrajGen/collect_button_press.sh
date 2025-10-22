conda activate omnicontrol
python -m sample.generate --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt --num_repetitions 1 --task button_press --output_dir sample/button_press_final --batch_size 128
conda activate env_isaaclab
cd sample/button_press_final1
rm -r *.pkl
cd ../button_press_final2
rm -r *.pkl
cd ../button_press_final
python3 retarget.py
cd ../button_press_final1
python3 refine_motions.py
cd ..
