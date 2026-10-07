"""Inference-only historical checkpoint audit on the development validation split."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import torch
from wordle_gpt.evaluation.evaluate_v2 import WordArgmaxPolicy, evaluate_model
from wordle_gpt.core.model import WordleGPT
from wordle_gpt.core.tokenizer import FEEDBACK_TO_SYMBOL
from wordle_gpt.core.tokenizer_v2 import encode, decode
from wordle_gpt.training.train import generate_constrained_guess
from wordle_gpt.distillation.train_soft_distillation import deterministic_execution, file_hash, write_json
from wordle_gpt.core.wordle import load_words, score_guess


def checkpoints():
    specifications = [
        ('hard-sft', 'runs/scaling-dev-1m/seed-0/fold-1/7.2m/checkpoints/best.pt'),
        ('anchored-dpo', 'runs/dpo-rescue-anchor/*/checkpoints/best.pt'),
        ('one-step-grpo', 'runs/grpo-dev-one-step/checkpoints/*.pt'),
        ('full-game-grpo', 'runs/grpo-games-dev/checkpoints/*.pt'),
        ('continuation-grpo', 'runs/grpo-continuations-beta-*/checkpoints/best.pt'),
        ('continuation-grpo', 'runs/grpo-token-lr-*/checkpoints/best*.pt'),
        ('state-conditioned-grpo', 'runs/grpo-dev-dense/checkpoints/*.pt'),
        ('state-conditioned-grpo', 'runs/grpo-information-dev/checkpoints/best*.pt'),
        ('state-conditioned-grpo', 'runs/grpo-information-dev/checkpoints/update-1000.pt'),
        ('soft-B', 'runs/soft-distillation-resumable/B-T0.25/checkpoints/best.pt'),
        ('soft-B', 'runs/soft-distillation-resumable/B-T0.25/resume.pt'),
        ('soft-B', 'runs/soft-distillation-dev/B-T0.25/checkpoints/best.pt'),
        ('soft-C-smoke-only', 'runs/soft-distillation-smoke/C-*/checkpoints/best.pt'),
    ]
    for family, pattern in specifications:
        for path in sorted(Path('.').glob(pattern)):
            yield family, path


def histories(results):
    visits = Counter()
    for game in results:
        prefix = encode('<P><G>')
        game['turns'] = []
        for guess in game['guesses']:
            visits[tuple(prefix)] += 1
            feedback = ''.join(FEEDBACK_TO_SYMBOL[x] for x in score_guess(game['secret'], guess))
            game['turns'].append({'prompt': decode(prefix), 'guess': guess, 'feedback': feedback})
            prefix += encode(guess + '<F>' + feedback + '<G>')
    return visits


def run(args):
    deterministic_execution()
    torch.set_num_threads(4)
    words = tuple(load_words())
    assert len(words) == 719
    secrets = json.loads(Path('data/wordle-development.json').read_text())['runs'][0]['validation']
    rows = []
    for family, checkpoint in checkpoints():
        identity = {'checkpoint': str(checkpoint), 'sha256': file_hash(checkpoint),
                    'words': words, 'secrets': secrets, 'audit_version': 1}
        identity = json.loads(json.dumps(identity))
        output = args.output_dir / (str(checkpoint).replace('/', '__') + '.json')
        if output.exists():
            report = json.loads(output.read_text())
            if report['identity'] != identity:
                raise ValueError(f'audit identity changed: {output}')
        else:
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            config = payload.get('model_config') or payload['best_checkpoint']['model_config']
            model = WordleGPT(**config).to(args.device).eval().requires_grad_(False)
            model.load_state_dict(payload['model_state_dict'])
            del payload
            games = {mode: asdict(evaluate_model(model, secrets, words, checkpoint=str(checkpoint), decode=mode))
                     for mode in ('constrained', 'word-argmax')}
            visits = {mode: histories(report['results']) for mode, report in games.items()}
            union = set().union(*visits.values())
            policy = WordArgmaxPolicy(model, words)
            allowed = frozenset(words)
            disagreements = {}
            with torch.inference_mode():
                for prompt in sorted(union):
                    token = generate_constrained_guess(model, list(prompt), allowed)
                    word = policy(list(prompt))
                    disagreements[prompt] = token != word
            disagreement = {'definition': 'Both decoders evaluated on each identical history in the unique union of visited histories',
                            'states': len(union), 'disagreements': sum(disagreements.values()),
                            'rate': sum(disagreements.values()) / len(union),
                            'visitation_weighted_rates': {mode: sum(disagreements[p] * n for p,n in counts.items()) / sum(counts.values())
                                                        for mode, counts in visits.items()}}
            assert file_hash(checkpoint) == identity['sha256']
            report = {'family': family, 'identity': identity, 'training_performed': False,
                      'games': games, 'decoder_disagreement': disagreement}
            write_json(output, report)
            del model, policy
            torch.cuda.empty_cache()
        row = {'family': family, 'checkpoint': str(checkpoint), 'report': str(output),
               'decoders': {mode: {k:v for k,v in g.items() if k != 'results'} for mode,g in report['games'].items()},
               'decoder_disagreement': report['decoder_disagreement']}
        rows.append(row)
        print(json.dumps({'family': family, 'checkpoint':str(checkpoint),
                          'wins': {m:g['wins'] for m,g in report['games'].items()},
                          'disagreement':report['decoder_disagreement']['rate']}), flush=True)
        write_json(args.output_dir / 'comparison.json', {'training_performed':False,
                   'split':'development validation', 'secrets':secrets, 'words':719,
                   'soft_C_caveat':'Only a smoke-run C checkpoint exists; no fully trained development C checkpoint.',
                   'checkpoints':rows, 'complete':False})
    result = json.loads((args.output_dir / 'comparison.json').read_text())
    result['complete'] = True
    write_json(args.output_dir / 'comparison.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('runs/checkpoint-decoder-audit'))
    parser.add_argument('--device', default='cuda')
    run(parser.parse_args())
