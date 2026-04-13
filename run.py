import yaml
import argparse

from src.utils.logger import Logger
from src.data.pipeline import DataCollectionPipeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['update', 'inference', 'summary'])
    parser.add_argument('--weights', help='Pretrained model for inference')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    logger = Logger(config).get_logger()
    pipeline = DataCollectionPipeline(config, logger)
    pipeline.run(args.mode, args.weights)