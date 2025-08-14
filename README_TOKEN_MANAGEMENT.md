# Secure Token Management

This repository has been configured to securely manage API tokens using environment variables loaded from a `.env` file. This approach prevents tokens from being committed to the repository, avoiding GitHub push protection issues.

## How It Works

1. API tokens are stored in a `.env` file in the root directory of the project
2. This file is listed in `.gitignore` so it won't be committed to the repository
3. Shell scripts load tokens from this file at runtime

## Setup Instructions

1. Create a `.env` file in the root directory of the project with your API tokens:
   ```
   WANDB_API_KEY=your_wandb_api_key_here
   HF_TOKEN=your_huggingface_token_here
   ```

2. Make sure the `.env` file is not committed to Git (it should already be in `.gitignore`)

3. Run your scripts as usual - they will automatically load the tokens from the `.env` file

## For New Scripts

When creating new scripts that need these tokens, use the following pattern:

```bash
# Load environment variables from .env file
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo "Error: .env file not found. Please create it with your API tokens."
  exit 1
fi

# Now you can use $WANDB_API_KEY and $HF_TOKEN in your script
wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN
```

## Security Notes

- Never commit the `.env` file to the repository
- When sharing the repository, make sure to exclude the `.env` file
- Each developer should create their own `.env` file with their personal tokens
