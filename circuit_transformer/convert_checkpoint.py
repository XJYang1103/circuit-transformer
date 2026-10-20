"""Convert the upstream TensorFlow checkpoint once, outside training servers.

Usage: python -m circuit_transformer.convert_checkpoint TF_PREFIX model.pt
"""
import argparse

from .model import CircuitTransformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='TensorFlow checkpoint prefix or .index path')
    parser.add_argument('output', help='Destination PyTorch .pt file')
    for name, default in [('num-inputs', 8), ('embedding-width', 512),
                          ('num-layers', 12), ('num-attention-heads', 8),
                          ('intermediate-size', 2048), ('max-tree-depth', 32)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--add-action-mask-to-input', action='store_true')
    args = vars(parser.parse_args())
    source, output = args.pop('source'), args.pop('output')
    model = CircuitTransformer(**args, mixed_precision=False, device='cpu')
    model.load(source)
    model.save(output)
    print(f'Converted all {len(model._transformer.state_dict())} model tensors to {output}')


if __name__ == '__main__':
    main()
