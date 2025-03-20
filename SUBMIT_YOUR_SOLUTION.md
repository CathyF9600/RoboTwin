# Submission Instructions

> YOUR_API_KEY will be provided through email upon registration.

Note:

- To ensure fairness, there is a frequency limit for creating submissions. There must be a three-day interval between two submissions.

First, download the submission tool:

- Windows AMD64: https://dist.bj.bcebos.com/windows-amd64/submit.exe
- Mac Arm64: https://dist.bj.bcebos.com/mac-arm64/submit
- Mac AMD64: https://dist.bj.bcebos.com/mac-amd64/submit
- Linux AMD64: https://dist.bj.bcebos.com/linux-amd64/submit

When you have downloaded the submission tool, you need to make it executable on Linux systems:

```bash
# Add executable permission to the file
chmod +x submit

# You can either:
# 1. Run it from the current directory
./submit

# 2. Or move it to a directory in your PATH for easier access
mv submit /usr/local/bin/   # Requires sudo permission
# or
mv submit ~/bin/           # Make sure ~/bin is in your PATH
```

Below examples assume you have added the Submit directory to your PATH.

The submission process consists of two steps: creating a submission and uploading files.
Please contact the competition organizers to obtain your APIKEY.

## 1. Create Submission

Create a new submission:

```bash
submit create --api-key YOUR_API_KEY --tasks TASK1,TASK2 # Get submission ID
```

supported tasks: `blocks_stack_hard` `bowls_stack` `dual_shoes_place` `put_bottles_dustbin` `empty_cup_place`
tasks should be separated by comma, e.g. `dual_shoes_place,empty_cup_place`

View all your submission records:

```bash
submit list --api-key YOUR_API_KEY  # View status and evaluation results of all submissions
```

## 2. Upload Files

Upload files from your local directory:

```bash
submit upload --api-key YOUR_API_KEY --submission-id YOUR_SUBMISSION_ID --dir /path/to/your/policy --checkpoint-dir /path/to/your/checkpoint
```

Notes:

- Submission ID is obtained when creating the submission
- The directory should contain all necessary files
- `--checkpoint-dir` is a required parameter that specifies the directory containing your model checkpoints
- Files in `dir/data` and `dir/checkpoints` will be automatically excluded from upload
- When using `--checkpoint-dir`, all files from the specified checkpoint directory will be uploaded to `checkpoints/[last-level-directory-name]/` path
- If the upload is interrupted, simply re-run the upload command. Previously uploaded files will not be uploaded again

## 3. View the Result and Leaderboard

After you submit your solution, you can view your result and score on the platform. The username and password for logging in will be provided through email upon registration.
