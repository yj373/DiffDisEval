import argparse
from huggingface_hub import snapshot_download

def parse_args():
    parser = argparse.ArgumentParser(description='download model')
    parser.add_argument(
        "--repo_id",
        type=str,
        default="Salesforce/blip-image-captioning-large",
        required=True,
        help="repo id of the target model",
    )
    parser.add_argument(
        "--destination",
        type=str,
        default="/dev/shm/alexJiang/source/",
        required=True,
        help="destination of the target model",
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    snapshot_download(repo_id=args.repo_id, local_dir=args.destination)