import sys
import os

# Set up the command line arguments for profiling
if __name__ == "__main__":
    # Override sys.argv to pass the profiler configuration via CLI
    sys.argv = [
        "main.py",
        "fit",
        "-c", "configs/ade20k_mask2former_semantic.yaml",
        "--root", "/home/valentin/workspaces/benchmark-vfm-ss/data",
        "--model.network.encoder_name", "vit_base_patch14_dinov2",
        # "--no_compile",
        "--model.freeze_encoder", "True",
        "--trainer.max_steps", "50",  # Only run 50 steps for profiling
        "--data.num_workers", "8",    # Reduce workers for cleaner profiling
        # PyTorch Profiler configuration via CLI
        "--trainer.profiler", "pytorch",
        # "--trainer.profiler.dirpath", "./profiler_logs",
        # "--trainer.profiler.filename", "mask2former_profile",
        # "--trainer.profiler.group_by_input_shapes", "True",
        # "--trainer.profiler.emit_nvtx", "False",
        # "--trainer.profiler.export_to_chrome", "True",
        # "--trainer.profiler.row_limit", "20",
        # "--trainer.profiler.record_module_names", "True",
    ]
    
    # Create profiler logs directory if it doesn't exist
    os.makedirs("./profiler_logs", exist_ok=True)
    
    # Import and run the main CLI
    from main import cli_main
    cli_main()
