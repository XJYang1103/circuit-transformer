"""Load unchanged TF references with only their required official NLP modules.

TF Models' package initializers import unrelated vision/tokenizer dependencies
that are not available on Windows. This loader bypasses those initializers; all
executed layers and math remain the installed, unmodified official sources.
"""
import importlib
import importlib.util
from pathlib import Path
import sys
import types


def load_reference():
    import official
    assert official.__file__ is not None
    root = Path(official.__file__).parent
    for name, relative in [('official.nlp.modeling', 'nlp/modeling'),
                           ('official.nlp.modeling.layers', 'nlp/modeling/layers'),
                           ('official.nlp.modeling.models', 'nlp/modeling/models')]:
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(root / relative)]
            sys.modules[name] = package
    layers = sys.modules['official.nlp.modeling.layers']
    for module, names in (
            ('on_device_embedding', ['OnDeviceEmbedding']),
            ('position_embedding', ['RelativePositionEmbedding']),
            ('transformer_encoder_block', ['TransformerEncoderBlock']),
            ('transformer', ['TransformerDecoderBlock'])):
        loaded = importlib.import_module('official.nlp.modeling.layers.' + module)
        for name in names:
            setattr(layers, name, getattr(loaded, name))
    spec = importlib.util.spec_from_file_location('tf_reference',
                                                  Path(__file__).with_name('tensorflow_transformer.py'))
    assert spec is not None and spec.loader is not None
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    return reference


def load_reference_circuit():
    reference = load_reference()
    stacks = importlib.import_module('official.nlp.modeling.models.seq2seq_transformer')
    source = Path(__file__).with_name('model.py').read_text(encoding='utf-8')
    # Only rebind imports to the reference dependencies; leave all method bodies unchanged.
    source = source.replace('from tensorflow_models import nlp', '')
    source = source.replace('from circuit_transformer.tensorflow_transformer import',
                            'from tf_reference import')
    sys.modules['tf_reference'] = reference
    module = types.ModuleType('circuit_reference')
    setattr(module, 'nlp', types.SimpleNamespace(models=stacks))
    exec(compile(source, str(Path(__file__).with_name('model.py')), 'exec'), module.__dict__)
    return module
